#! /usr/bin/env python3
import re
from typing import List, Any

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import pandas as pd
import os
import sys
import logging
import time
import numpy as np
import inspect
import Levenshtein
from tqdm import tqdm

sys.path.append("../../baseline/utils")
from file_utils import mkdir

sys.path.append("../../..")
from trec_car_tools.python3.trec_car.read_data_own import iter_pages, iter_paragraphs
from trec_car_tools.python3.trec_car.read_data_own import Para as TypePara
from trec_car_tools.python3.trec_car.read_data_own import Section as TypeSection

logging.basicConfig(level=logging.INFO)

global log_error_file
log_error_file = "./log_errors.txt"


def get_text(page, tokenizer):
    text = ""
    for t in page.get_text_with_headings():
        text += (t)
    # text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"^\n", "", text)
    text = text.split("\n")

    text_id = {"id": [], "text": []}
    for t in text:
        if not re.search(r"\w{3,}", t):
            continue
        match = re.match(r"^\[ID-I\]\d+\[ID-O\]", t)
        if match:
            index = re.sub(r"(?:\[ID-I\]|\[ID-O\])", "", match.group())
            text = t[match.end():]
            if not re.search(r"\w{3,}", text):
                continue
            text_id["id"].append(str(index))
            text_id["text"].append(text)
        else:
            text_id["id"].append(-1)
            text_id["text"].append(t)
    text_id = pd.DataFrame(text_id)
    text_id["text_w/o_heading"] = text_id["text"].apply(
        lambda paragraph: "" if re.match("^\[h\d\].+\[h\d\]$", paragraph) else paragraph
    )
    text_id["text_w_heading_first_sentence"] = text_id["text"].apply(
        lambda paragraph: tokenizer.tokenize(paragraph)[0]
    )
    text_id["text_w/o_heading_first_sentence"] = text_id["text_w/o_heading"].apply(
        lambda paragraph: tokenizer.tokenize(paragraph)[0] if paragraph else ""
    )
    text_id = text_id.rename(columns={"text": "text_w_heading_all_passage",
                                      "text_w/o_heading": "text_w/o_heading_all_passage"})
    text_no_heading = text_id[["id",
                               "text_w/o_heading_all_passage",
                               "text_w/o_heading_first_sentence"]].query("id != -1").reset_index(drop=True)
    text_wi_heading = text_id[["id",
                               "text_w_heading_all_passage",
                               "text_w_heading_first_sentence"]].reset_index(drop=True)
    text_wi_heading = get_span(text_wi_heading)
    text_no_heading = get_span(text_no_heading)
    text = {"text_w_heading_all_passage": "\n".join(text_wi_heading["text_w_heading_all_passage"].to_list()),
            "text_w/o_heading_all_passage": "\n".join(text_no_heading["text_w/o_heading_all_passage"].to_list()),
            "text_w_heading_first_sentence": "\n".join(text_wi_heading["text_w_heading_first_sentence"].to_list()),
            "text_w/o_heading_first_sentence": "\n".join(text_no_heading["text_w/o_heading_first_sentence"].to_list())}

    span = dict()
    for col in text.keys():
        if "_w_" in col:
            span[
                col.replace("text_", "span_")
            ] = text_wi_heading[col.replace("text_", "span_")][text_wi_heading["id"] != -1]
        elif "_w/o_" in col:
            span[
                col.replace("text_", "span_")
            ] = text_no_heading[col.replace("text_", "span_")][text_no_heading["id"] != -1]

    ids = text_id["id"][text_id["id"] != -1].to_list()
    df = {"id": ids}

    for k_text, k_span in zip(text.keys(), span.keys()):
        df[k_text] = text[k_text]
        df[k_span] = span[k_span].to_list()

    return df


def get_span(text):
    text_col = text.columns[["id" not in col for col in text.columns]].to_list()
    span = {col: [] for col in text_col}
    text_len = dict()
    for col in text_col:
        text_len[col] = text[col].apply(lambda paragraph: len(paragraph))
    for i in range(len(text)):
        for col in text_col:
            start = sum(text_len[col][:i]) + i
            end = start + text_len[col][i:i + 1].item()
            span[col].append([start, end])
    for col in text_col:
        t = "\n".join(text[col].to_list())
        for i in range(len(text)):
            assert t[span[col][i][0]:span[col][i][1]] == text[col][i:i + 1].item()
        text[col.replace("text_", "span_")] = span[col]
    return text


def construct_corpus(path_file: str, tokenizer):
    """
    :param path_file: path_file: path to the *.cbor file to process iter_paragraphs
    :return: df: DataFrame with columns :
        - text : *text from the corpus*
        - id : *corresponding id*
    """
    corpus = {"id": [], "all_passage": [], "first_sentence": []}
    with open(path_file, "rb") as file_corpus:
        corpus_cbor = list(iter_paragraphs(file_corpus))

    logging.info(f"corpus size : {sys.getsizeof(corpus_cbor)} octets")

    for paragraph in tqdm(corpus_cbor, miniters=len(corpus_cbor) // 10, mininterval=60, maxinterval=180):
        # if re.search("\w{3,}", str(paragraph.get_text())):
        #     corpus["id"].append(paragraph.para_id)
        #     corpus["text"].append(paragraph.get_text())
        # else :
        #     continue
        paragraph_text = paragraph.get_text().replace("\n", "")
        match = re.match(r"^\[ID-I\]\d+\[ID-O\]", paragraph_text)
        # match = str(paragraph).split("[ID-O]")
        # if len(match) == 2:
        if match is not None:
            # index = match[0].replace("[ID-I]", "")
            # text = match[1]
            index = re.sub(r"(?:\[ID-I\]|\[ID-O\])", "", match.group())
            text = paragraph_text[match.end():]
            if not re.search(r"\w{3,}", text):
                continue
            corpus["id"].append(str(index))
            corpus["all_passage"].append(text)
            corpus["first_sentence"].append(tokenizer.tokenize(text)[0])

    corpus = pd.DataFrame.from_dict(corpus)
    return corpus


def construct_article(path_file: str,
                      tokenizer):
    """
    :param path_file: path to the *.cbor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :param print_error: whether to print the average missing index or not.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_<type>                                   : *gold_output*
            - paragraphs_id_<type>                           : *ids of the paragraphs' gold output in the corpus*
            - paragraphs_span_<type>                         : *span location in gold output of the paragraphs
                                                                from corpus*
            NOTE : paragraphs_id[k] corresponds to paragraphs_span[k]*
            <type> in [
                    w_heading_all_passage,                  # with heading and entire page
                    w/o_heading_all_passage,                # without heading and entire page
                    w_heading_first_sentence,  # with heading and first sentence of each paragraph
                    w/o_heading_first_sentence,  # without heading and first sentence of each paragraph
    """
    df = []
    df_skipped = []
    count_pages = 0
    skipped_pages = 0
    no_ids = 0
    with open(path_file, 'rb') as file:
        pages = list(iter_pages(file))
    for page in tqdm(pages, miniters=len(pages) // 10, mininterval=60, maxinterval=180):
        text = get_text(page, tokenizer)
        text["query"] = page.page_name
        if len(text["id"]) > 0:
            if page.child_sections:
                count_pages += 1
                text["outline"] = [r"///".join([str(section.heading) for section in section_path])
                                   for section_path in page.flat_headings_list()]
                df.append(text)
            else:
                skipped_pages += 1
                df_skipped.append(text)
        else:
            no_ids += 1

    df = pd.DataFrame(df)
    df = df[[
        'query',
        'outline',
        'text_w_heading_all_passage',
        'span_w_heading_all_passage',
        'text_w/o_heading_all_passage',
        'span_w/o_heading_all_passage',
        'text_w_heading_first_sentence',
        'span_w_heading_first_sentence',
        'text_w/o_heading_first_sentence',
        'span_w/o_heading_first_sentence',
        'id'
    ]]

    if df_skipped != []:
        df_skipped = pd.DataFrame(df_skipped)
        df_skipped = df_skipped[[
            'query',
            'text_w/o_heading_all_passage',
            'span_w/o_heading_all_passage',
            'text_w/o_heading_first_sentence',
            'span_w/o_heading_first_sentence',
            'id'
        ]]
        df_skipped = df_skipped.rename(columns={'text_w/o_heading_all_passage': 'text_all_passage',
                                                'span_w/o_heading_all_passage': 'span_all_passage',
                                                'text_w/o_first_sentence': 'text_first_sentence',
                                                'span_w/o_first_sentence': 'span_first_sentence'})

    logging.info(f"Constructed pages : {str(len(df) + len(df_skipped)).rjust(7)} "
                 f" ({(len(df) + len(df_skipped)) / count_pages * 100:.1f}%)".ljust(42) + "\t;\t" +
                 f"Missed pages : {str(count_pages - (len(df) + len(df_skipped))).rjust(7)} "
                 f" ({(count_pages - (len(df) + len(df_skipped))) / count_pages * 100:.1f}%)".ljust(42) + "\t;\t" +
                 f"Skipped pages : {str(skipped_pages).rjust(7)} " + "\t;\t" +
                 f"no ids : {str(no_ids).rjust(7)} ")
    return df, df_skipped


def construct_section(path_file: str,
                      tokenizer):
    """
    :param path_file: path to the *.cbor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_<type>                                   : *gold_output*
            - paragraphs_id_<type>                           : *ids of the paragraphs' gold output in the corpus*
            - paragraphs_span_<type>                         : *span location in gold output of the paragraphs
                                                                from corpus*
            NOTE : paragraphs_id[k] corresponds to paragraphs_span[k]*
            <type> in [
                    w_heading_all_passage,                  # with heading and entire section
                    w/o_heading_all_passage,                # without heading and entire section
                    w_heading_first_sentence,  # with heading and first sentence of each paragraph
                    w/o_heading_first_sentence,  # without heading and first sentence of each paragraph


    """
    df = []
    df_skipped = []
    count_sections = 0
    skipped_sections = 0
    no_ids = 0
    with open(path_file, 'rb') as file:
        pages = list(iter_pages(file))
    for page in tqdm(pages, miniters=len(pages) // 10, mininterval=60, maxinterval=180):
        if page.child_sections:
            for section in page.child_sections:
                text = get_text(section, tokenizer)
                text["query"] = page.page_name + r"///" + section.heading
                count_sections += 1
                if len(text["id"]) > 0:
                    if section.child_sections:
                        text["outline"] = [r"///".join([page.page_name] + [str(s.heading)
                                                                           for s in
                                                                           sectionpath[sectionpath.index(section)
                                                                                       + 1:len(sectionpath)]])
                                           for sectionpath in page.flat_headings_list() if section in sectionpath]
                        df.append(text)
                    else:
                        skipped_sections += 1
                        df_skipped.append(text)
                else:
                    no_ids += 1

    df = pd.DataFrame(df)
    df = df[[
        'query',
        'outline',
        'text_w_heading_all_passage',
        'span_w_heading_all_passage',
        'text_w/o_heading_all_passage',
        'span_w/o_heading_all_passage',
        'text_w_heading_first_sentence',
        'span_w_heading_first_sentence',
        'text_w/o_heading_first_sentence',
        'span_w/o_heading_first_sentence',
        'id'
    ]]

    if df_skipped != []:
        df_skipped = pd.DataFrame(df_skipped)
        df_skipped = df_skipped[[
            'query',
            'text_w/o_heading_all_passage',
            'span_w/o_heading_all_passage',
            'text_w/o_heading_first_sentence',
            'span_w/o_heading_first_sentence',
            'id'
        ]]
        df_skipped = df_skipped.rename(columns={'text_w/o_heading_all_passage': 'text_all_passage',
                                                'span_w/o_heading_all_passage': 'span_all_passage',
                                                'text_w/o_first_sentence': 'text_first_sentence',
                                                'span_w/o_first_sentence': 'span_first_sentence'})

    logging.info(f"Constructed sections : {str(len(df)).rjust(7)} "
                 f" ({(len(df) + len(df_skipped)) / count_sections * 100:.1f}%)".ljust(42) + "\t;\t" +
                 f"Missed sections : {str(count_sections - (len(df) + len(df_skipped))).rjust(7)} "
                 f" ({(count_sections - (len(df) + len(df_skipped))) / count_sections * 100:.1f}%)".ljust(42) + "\t;\t" +
                 f"Skipped sections : {str(skipped_sections).rjust(7)} " + "\t;\t" +
                 f"no ids : {str(no_ids).rjust(7)} ")
    return df, df_skipped


def main(path_file: str,
         path_output_folder: str,
         save_as: str = "json"):
    """
    :param path_file: path to the *.cboor file to process
    :param path_output_folder: path to the save location. In addition, a sub-folder for the fold will be created.
    :return:
    """
    t_start = time.time()
    logging.info(f"Start processing {path_file}.")

    logging.info("Parse path_file for 'data type_train_test' (and 'fold' if train data).")
    try:
        type_train_test = re.findall(r"train|test", path_file)[0]
    except IndexError:
        type_train_test = "corpus"
    logging.info(f"\tfile type : {type_train_test}.")
    paragraphs_only = True if re.findall(r"paragraph", path_file) else False
    logging.info(f"\tfile contains {'paragraphs corpus' if paragraphs_only else 'wikipedia pages'}.")

    if type_train_test == "train":
        try:
            fold = int(re.findall(r"\d+", re.findall(r"(fold-\d|fold\d)", path_file)[0])[0])
            path_output_folder = os.path.join(path_output_folder, f"fold-{fold}")
        except IndexError:
            path_output_folder = os.path.join(path_output_folder, f"all-")
    elif type_train_test == "test":
        path_output_folder = os.path.join(path_output_folder, f"test")
    elif paragraphs_only:
        path_output_folder = os.path.join(path_output_folder, f"corpus")

    logging.info('Check if output path exists or create it.')
    mkdir(path_output_folder)

    logging.info("Get tokenizer.")
    punkt_param = PunktParameters()
    abbreviation = ["A", "a", "A.B.C", "a.b.c", "A.C.A.D", "a.c.a.d", "A.K.A", "a.k.a", "A.L", "a.l", "A.M", "a.m",
                    "A.R.R", "a.r.r", "A.S.S.N", "a.s.s.n", "ABC", "abc", "ACAD", "acad", "AKA", "aka", "AL", "al",
                    "AM", "am", "ARR", "arr", "ASSN", "assn", "B.L.V.D", "b.l.v.d", "BLVD", "blvd", "C", "c", "C.A",
                    "c.a", "C.A.P.T", "c.a.p.t", "C.F", "c.f", "C.I.A", "c.i.a", "C.M.D.R", "c.m.d.r", "C.O", "c.o",
                    "C.O.L", "c.o.l", "C.O.L.L", "c.o.l.l", "C.O.R.P", "c.o.r.p", "C.P.L", "c.p.l", "CA", "ca",
                    "CAPT", "capt", "CF", "cf", "CIA", "cia", "CMDR", "cmdr", "CO", "co", "COL", "col", "COLL",
                    "coll", "CORP", "corp", "CPL", "cpl", "D.B.A", "d.b.a", "D.C", "d.c", "D.R", "d.r", "DBA",
                    "dba", "DC", "dc", "DR", "dr", "E", "e", "E.D", "e.d", "E.D.S", "e.d.s", "E.G", "e.g", "E.T.C",
                    "e.t.c", "ED", "ed", "EDS", "eds", "EG", "eg", "ETC", "etc", "F.E.A.T", "f.e.a.t", "F.I.G",
                    "f.i.g", "F.L", "f.l", "F.T", "f.t", "F.W.Y", "f.w.y", "FEAT", "feat", "FIG", "fig", "FL", "fl",
                    "FT", "ft", "FWY", "fwy", "G.E.N", "g.e.n", "GEN", "gen", "H.O.N", "h.o.n", "H.W.Y", "h.w.y",
                    "HON", "hon", "HWY", "hwy", "I.E", "i.e", "I.N.C", "i.n.c", "I.N.S.T", "i.n.s.t", "I.T", "i.t",
                    "IE", "ie", "INC", "inc", "INST", "inst", "IT", "it", "J.R", "j.r", "JR", "jr", "L.T", "l.t",
                    "L.T.D", "l.t.d", "LT", "lt", "LTD", "ltd", "M", "m", "M.A.J", "m.a.j", "M.F.G", "m.f.g",
                    "M.I.S.S", "m.i.s.s", "M.O.N.S", "m.o.n.s", "M.S.E", "m.s.e", "M.S.G.R", "m.s.g.r", "M.S.G.T",
                    "m.s.g.t", "M.T", "m.t", "M.T.N", "m.t.n", "MAJ", "maj", "MFG", "mfg", "MISS", "miss", "MONS",
                    "mons", "MSE", "mse", "MSGR", "msgr", "MSGT", "msgt", "MT", "mt", "MTN", "mtn", "N", "n", "N.A",
                    "n.a", "N.A.T", "n.a.t", "N.E", "n.e", "N.W", "n.w", "NA", "na", "NAT", "nat", "NE", "ne", "NW",
                    "nw", "P.H.D", "p.h.d", "P.M", "p.m", "P.R", "p.r", "P.R.O.F", "p.r.o.f", "P.U.B", "p.u.b",
                    "P.U.B.G", "p.u.b.g", "P.U.B.S", "p.u.b.s", "P.V.T", "p.v.t", "PHD", "phd", "PM", "pm", "PR",
                    "pr", "PROF", "prof", "PUB", "pub", "PUBG", "pubg", "PUBS", "pubs", "PVT", "pvt", "R.D", "r.d",
                    "R.E.V", "r.e.v", "R.T", "r.t", "RD", "rd", "REV", "rev", "RT", "rt", "S", "s", "S.E", "s.e",
                    "S.G.T", "s.g.t", "S.Q", "s.q", "S.R", "s.r", "S.S.G.T", "s.s.g.t", "S.T", "s.t", "S.T.E",
                    "s.t.e", "S.W", "s.w", "SE", "se", "SGT", "sgt", "SQ", "sq", "SR", "sr", "SSGT", "ssgt", "ST",
                    "st", "STE", "ste", "SW", "sw", "T.R.A.D", "t.r.a.d", "T.S.G.T", "t.s.g.t", "TRAD", "trad",
                    "TSGT", "tsgt", "U", "u", "U.N.I", "u.n.i", "U.N.I.V", "u.n.i.v", "U.S", "u.s", "U.S.A",
                    "u.s.a", "UNI", "uni", "UNIV", "univ", "US", "us", "USA", "usa", "V", "v", "V.I.Z", "v.i.z",
                    "V.S", "v.s", "VIZ", "viz", "VS", "vs", "W", "w"]
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)

    if paragraphs_only:  # the file is a corpus
        logging.info("construct corpus dataset")
        df_corpus = construct_corpus(path_file, tokenizer)
        logging.info("save dataset")
        if save_as == "scv":
            logging.info(str(os.path.join(path_output_folder, f"corpus_{type_train_test}.csv")))
            df_corpus.to_csv(os.path.join(path_output_folder, f"corpus_{type_train_test}.csv"), index=False)
        elif save_as == "json":
            logging.info(str(os.path.join(path_output_folder, f"corpus_{type_train_test}.json")))
            df_corpus.to_json(os.path.join(path_output_folder, f"corpus_{type_train_test}.json"), indent=True)
        logging.info(f"DONE. Elepased time : {time.time() - t_start:.2f}s.\n\n")
        del df_corpus
    else:
        ## IDS AND SPANS
        if save_as == "csv":
            file_title = f"{type_train_test}.csv"
        elif save_as == "json":
            file_title = f"{type_train_test}.json"
        else:
            file_title = None
        ## ARTICLE
        logging.info("construct article dataset")
        df_article, df_article_skipped = construct_article(path_file,
                                                           tokenizer)
        logging.info(f"\tsave dataset at {os.path.join(path_output_folder, f'articles_{file_title}')}")
        if save_as == "csv":
            df_article.to_csv(os.path.join(path_output_folder, f"articles_{file_title}"), index=False)
        elif save_as == "json":
            df_article.to_json(os.path.join(path_output_folder, f"articles_{file_title}"), indent=True)
        del (df_article)
        ### skipped ones
        if type(df_article_skipped) is not list:
            logging.info(f"\tsave dataset at {os.path.join(path_output_folder, f'skipped_articles_{file_title}')}")
            if save_as == "csv":
                df_article_skipped.to_csv(os.path.join(path_output_folder, f"skipped_articles_{file_title}"),
                                          index=False)
            elif save_as == "json":
                df_article_skipped.to_json(os.path.join(path_output_folder, f"skipped_articles_{file_title}"),
                                           indent=True)
            del (df_article_skipped)
        # SECTION
        logging.info("construct section dataset")
        df_section, df_section_skipped = construct_section(path_file,
                                                           tokenizer)
        logging.info(f"\tsave dataset at {os.path.join(path_output_folder, f'sections_{file_title}')}")
        if save_as == "csv":
            df_section.to_csv(os.path.join(path_output_folder, f"sections_{file_title}"), index=False)
        elif save_as == "json":
            df_section.to_json(os.path.join(path_output_folder, f"sections_{file_title}"), indent=True)
        del (df_section)
        ### skipped ones
        if type(df_section_skipped) is not list:
            logging.info(f"\tsave dataset at {os.path.join(path_output_folder, f'skipped_sections_{file_title}')}")
            if save_as == "csv":
                df_section_skipped.to_csv(os.path.join(path_output_folder, f"skipped_sections_{file_title}"),
                                          index=False)
            elif save_as == "json":
                df_section_skipped.to_json(os.path.join(path_output_folder, f"skipped_sections_{file_title}"),
                                           indent=True)
            del (df_section_skipped)

        logging.info(f"DONE. Elapsed time : {time.time() - t_start:.2f}s.\n\n")


if __name__ == "__main__":
    # train subset
    logging.info("\n===== SUB- DATASET =====")
    ## folds
    for i in range(0, 5):
        main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor",
             path_output_folder="../../../data-subset_pre_processed")
        ## corpus
        main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor-paragraphs.cbor",
             path_output_folder="../../../data-subset_pre_processed")
    # all
    main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-train/train.pages.cbor",
         path_output_folder="../../../data-subset_pre_processed")
    ### corpus
    main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-train/train.pages.cbor-paragraphs.cbor",
         path_output_folder="../../../data-subset_pre_processed")
    # test subset
    main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor",
         path_output_folder="../../../data-subset_pre_processed")
    ### corpus
    main(path_file=f"../../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor-paragraphs.cbor",
         path_output_folder="../../../data-subset_pre_processed")

    # ONLY ON OSIRIM
    logging.info("\n===== FULL DATASET =====")
    # test set
    main(path_file=f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/test200/train.test200.cbor",
         path_output_folder="../../../data-set_pre_processed")
    ### corpus
    main(path_file=f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/test200/train.test200.cbor.paragraphs",
         path_output_folder="../../../data-set_pre_processed")
    # train set
    ## folds
    for i in range(0, 5):
        main(path_file=f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/train/train.fold{i}.cbor",
             path_output_folder="../../../data-set_pre_processed")
        ### corpus
        main(path_file=f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/train/train.fold{i}.cbor.paragraphs",
             path_output_folder="../../../data-set_pre_processed")

    # ### entire corpus  # too large, cannot run (> 10Go)
    # main(path_file=f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/paragraphcorpus/paragraphcorpus.cbor",
    #      path_output_folder="../../../data-set_pre_processed")
