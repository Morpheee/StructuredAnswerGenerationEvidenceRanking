#! /usr/bin/env python3
import re
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

sys.path.append("../../.")
from trec_car_tools.python3.trec_car.read_data import iter_pages, iter_paragraphs
from trec_car_tools.python3.trec_car.read_data import Para as TypePara
from trec_car_tools.python3.trec_car.read_data import Section as TypeSection

logging.basicConfig(level=logging.INFO)


def clean_df(df):
    for col in df.columns:
        if "text" in col:
            df[col] = df[col].apply(clean_text)
    return df


def leven_dist(str_a, str_b, distance="relative"):
    if type(str_a) == str:
        absolute_distance = Levenshtein.distance(str_a, str_b)
        relative_distance = absolute_distance / (len(str_a) + len(str_b))
    else:  # if type(str_a) == list or type(str_a) == np.ndarray:
        absolute_distance = []
        relative_distance = []
        for str_a_i in str_a:
            absolute_distance.append(Levenshtein.distance(str_a_i, str_b))
            relative_distance.append(absolute_distance[-1] / (len(str_a_i) + len(str_b)))
        absolute_distance = np.array(absolute_distance)
        relative_distance = np.array(relative_distance)
    if distance == "relative":
        return relative_distance
    elif distance == "absolute":
        return absolute_distance


def leven_sim(str_a, str_b, threshold=0.005):
    return leven_dist(str_a, str_b) < threshold


def clean_text(text):
    # remove asterists
    text = re.sub(r"\*", '', text)
    # remove multiple line break
    text = re.sub(r"\n{2,}", "\n", text)
    # remove text between brackets after text between square brackets (i.e. references)
    text = re.sub(r"""\]\((?:\s|[a-zA-Z0-9!@#$%^&*_+\(\-=\[\]{};':"\\|,.<>\/?])+\)""", "]", text)
    text = re.sub(r"""\](?:\s|[a-zA-Z0-9!@#$%^&*_+\)\-=\[\]{};':"\\|,.<>\/?])+\)""", "]", text)
    # remove square brackets around references
    refs = re.finditer(r"""\[(?:\s|[a-zA-Z0-9!@#$%^&*_+\-=\[\]{};':"\\|,.<>\/?])+\]""", text)
    for ref in list(refs)[::-1]:
        text = text[:ref.start()] + re.sub(r"\[|\]", "", ref.group()) + text[ref.end():]
    # remove multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    # remove space, if there is, before a point
    text = re.sub(r"\s\.", "../..", text)
    # remove space, if there is, before a coma
    text = re.sub(r"\s\,", ",", text)
    # add space after a full stop at the end of a sentence (sometimes, due to references, there is no space between
    # two sentences.)
    results = re.finditer(r"[a-z]\.[A-Z]", text)
    for result in list(results)[::-1]:
        text = text[:result.start()] + re.sub(r"../..", ". ", result.group()) + text[result.end():]
    return text


def construct_corpus(path_file: str):
    """
    :param path_file: path_file: path to the *.cboor file to prociter_paragraphsess
    :return: df: DataFrame with columns :
        - text : *text from the corpus*
        - id : *corresponding id*
    """
    corpus = {"id": [], "text": []}
    with open(path_file, "rb") as file_corpus:
        for paragraph in tqdm(list(iter_paragraphs(file_corpus))):
            if paragraph.para_id not in corpus["id"] and \
                    not any(leven_sim(corpus["text"], paragraph.get_text())):
                corpus["id"].append(paragraph.para_id)
                corpus["text"].append(paragraph.get_text())
    corpus = pd.DataFrame.from_dict(corpus)
    return corpus


def get_paragraphs(page):
    def recurs(passage):
        if type(passage) == TypePara:
            yield passage.paragraph.para_id, passage.paragraph.get_text()
        elif type(passage) == TypeSection:
            for section in passage:
                for para_info in recurs(section):
                    yield para_info
        else:
            try:
                yield passage.paragraph.para_id, passage.paragraph.get_text()
            except AttributeError:
                logging.debug(f"ParagraphId unsupported type : {type(passage)}")

    para_ids = []
    page_paragraphs = []
    for page_skeleton in page.skeleton:
        for para_id, paragraph in recurs(page_skeleton):
            if para_id in para_ids or any(leven_sim(page_paragraphs, paragraph)):
                print("already there :",paragraph[:150])
                continue
            else:
                para_ids.append(para_id)
                page_paragraphs.append(paragraph)
    return {"para_ids": para_ids, "page_paragraphs": np.array(page_paragraphs)}


def get_intro(page, tokenizer):
    # handle page intro
    first_section = page.child_sections[0].get_text()
    intro_all_passage_list = []
    intro_first_sentence_list = []
    for passage_skeleton in page.skeleton:
        passage = passage_skeleton.get_text()
        if passage == first_section:
            break  # stop here, it means we reached the end of the intro
        else:
            intro_all_passage_list.append(passage)
            try:
                first_sentence = tokenizer.tokenize(passage)[0]
            except IndexError:
                first_sentence = ""
            intro_first_sentence_list.append(first_sentence)
    intro_all_passage = "\n".join(intro_all_passage_list)
    intro_first_sentence_by_paragraph = "\n".join(intro_first_sentence_list)

    intro = {"text_w_heading_first_sentence_by_paragraph": intro_first_sentence_by_paragraph,
             "text_w_heading_all_passage": intro_all_passage,
             "text_w/o_heading_first_sentence_by_paragraph": intro_first_sentence_by_paragraph,
             "text_w/o_heading_all_passage": intro_all_passage}

    return intro


def get_intro_section(section, tokenizer):
    # handle section intro
    if len(section.children) > 1:
        child0, child1 = section.children[:2]
        if child0.get_text_with_headings(include_heading=True, level=1)[:1] != "=" and \
                child1.get_text_with_headings(include_heading=True, level=1)[:1] == "=":
            intro_all_passage = child0.get_text_with_headings(include_heading=False, level=1)
            intro_first_sentence = tokenizer.tokenize(intro_all_passage)[0]
        else:
            intro_all_passage = ""
            intro_first_sentence = ""
    else:
        intro_all_passage = ""
        intro_first_sentence = ""

    intro = {"text_w_heading_first_sentence_by_paragraph": intro_first_sentence,
             "text_w_heading_all_passage": intro_all_passage,
             "text_w/o_heading_first_sentence_by_paragraph": intro_first_sentence,
             "text_w/o_heading_all_passage": intro_all_passage}

    return intro


def get_text(page, texts, tokenizer):
    def recurs(section, headings=False, level=1):
        if len(section.child_sections) == 0:
            if headings:
                passage = section.get_text_with_headings(include_heading=True, level=level)
            else:
                passage = section.get_text_with_headings(include_heading=False, level=level)
            for paragraph in passage.split("\n"):
                if paragraph != '':
                    yield tokenizer.tokenize(paragraph)[0]
                else:
                    continue
        else:
            if headings:
                yield "=" * level + str(section.heading) + "=" * level
            if len(section.children) > 1:
                child0, child1 = section.children[:2]
                if child0.get_text_with_headings(include_heading=True, level=level)[:1] != "=" and \
                        child1.get_text_with_headings(include_heading=True, level=level)[:1] == "=":
                    yield tokenizer.tokenize(child0.get_text_with_headings(include_heading=headings, level=level))[0]
            for subsection in section.child_sections:
                for passage in recurs(subsection, headings, level=level + 1):
                    yield passage

    for key in texts.keys():
        for section in page.child_sections:
            text = ""
            try:
                if "first" in key:
                    if "_w_" in key:
                        for passage in recurs(section, headings=True):
                            text += "\n" + passage
                    elif "_w/o_" in key:
                        for passage in recurs(section):
                            text += "\n" + passage
                elif "all" in key and "_w_" in key:
                    text += "\n" + section.get_text_with_headings(include_heading=True)
                elif "all" in key and "_w/o_" in key:
                    text += "\n" + section.get_text_with_headings(include_heading=False)
            except:
                continue
            finally:
                texts[key] += text
    return texts


def get_paragraphs_id_and_span(texts, tokenizer, para_ids, para_text, passage_matching_error: list = None):
    paragraphs_span = dict()
    for key in texts.keys():
        if "text" in key:
            paragraphs_span[key] = []

    paragraphs_index = dict()
    for key in texts.keys():
        if "text" in key:
            paragraphs_index[key] = []

    para_text_first_sent_tmp = [tokenizer.tokenize(psg)[0] for psg in para_text]
    para_text_first_sent = []
    # eliminate multiple iterations
    for paragraph in para_text_first_sent_tmp :
        if not any(leven_sim(para_text_first_sent, paragraph)) :
            para_text_first_sent.append(paragraph)

    for key in texts.keys():
        if "text" in key:
            text_split = np.array(texts[key].split("\n"))
            for index, passage in zip(para_ids, [para_text_first_sent, para_text][0 if "first" in key else 1]):
                try:
                    paragraphs_index_tmp = []
                    paragraphs_span_tmp = []
                    if sum(leven_sim(text_split, passage)) > 1:
                        logging.warning(f"Paragraph matches {sum(leven_sim(text_split, passage))} passages."
                                        " -> expected if passage appears multiple time in page."
                                        f" Page : {texts['query']}\n"
                                        f"Issued passage : '{passage[:150]}{' [...]' if len(passage)> 150 else ''}'")
                    for location in np.where(leven_sim(text_split, passage))[0]:
                        paragraphs_index_tmp.append(index)
                        span_start = sum([len(psg) for psg in text_split[:location]]) + location
                        span_end = span_start + len(text_split[location])
                        paragraphs_span_tmp.append([span_start, span_end])
                    paragraphs_index[key] += paragraphs_index_tmp
                    paragraphs_span[key] += paragraphs_span_tmp
                except TypeError:
                    logging.warning("TypeError when identifying passage.")
                    continue
            if passage_matching_error is not None:
                passage_matching_error.append(abs(len(para_ids) - len(paragraphs_index[key])) / len(para_ids))
    for key in paragraphs_span.keys():
        if "text" in key:
            key_span = key.replace("text", "paragraph_span")
            key_id = key.replace("text", "paragraph_id")
            texts[key_span] = paragraphs_span[key]
            texts[key_id] = paragraphs_index[key]

    return texts


def construct_article(path_file: str, tokenizer, print_error=True):
    """
    :param path_file: path to the *.cbor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :param print_error: whether to print the average missing index or not.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_<type>                                   : *gold_output*
            - paragraph_id_<type>                           : *ids of the paragraphs' gold output in the corpus*
            - paragraph_span_<type>                         : *span location in gold output of the paragraphs
                                                                from corpus*
            NOTE : paragraph_id[k] corresponds to paragraph_span[k]*
            <type> in [
                    w_heading_all_passage,                  # with heading and entire page
                    w/o_heading_all_passage,                # without heading and entire page
                    w_heading_first_sentence_by_paragraph,  # with heading and first sentence of each paragraph
                    w/o_heading_first_sentence_by_paragraph,  # without heading and first sentence of each paragraph
    """
    df = []
    if print_error:
        passage_matching_error = []
    else:
        passage_matching_error = None
    with open(path_file, 'rb') as file:
        for page in tqdm(list(iter_pages(file))):
            if page.child_sections:
                para_ids, para_text = get_paragraphs(page).values()
                texts = get_intro(page, tokenizer)
                texts = get_text(page, texts, tokenizer)
                texts["query"] = page.page_name
                texts["outline"] = ["/".join([str(section.heading) for section in section_path])
                                    for section_path in page.flat_headings_list()]
                texts = get_paragraphs_id_and_span(texts, tokenizer, para_ids, para_text, passage_matching_error)
                df.append(texts)
    df = pd.DataFrame(df)
    df = df[["query", "outline",
             "text_w_heading_all_passage", "paragraph_span_w_heading_all_passage",
             "paragraph_id_w_heading_all_passage",
             "text_w/o_heading_all_passage", "paragraph_span_w/o_heading_all_passage",
             "paragraph_id_w/o_heading_all_passage",
             "text_w_heading_first_sentence_by_paragraph", "paragraph_span_w_heading_first_sentence_by_paragraph",
             "paragraph_id_w_heading_first_sentence_by_paragraph",
             "text_w/o_heading_first_sentence_by_paragraph", "paragraph_span_w/o_heading_first_sentence_by_paragraph",
             "paragraph_id_w/o_heading_first_sentence_by_paragraph"]]
    if print_error:
        passage_matching_error = sum(passage_matching_error) / len(passage_matching_error)
        logging.info(f"average passage_matching_error : {passage_matching_error * 100:.1f}% missing matches")
    return df


def construct_section(path_file: str, tokenizer):
    """
    :param path_file: path to the *.cbor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :param print_error: whether to print the average missing index or not.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_<type>                                   : *gold_output*
            - paragraph_id_<type>                           : *ids of the paragraphs' gold output in the corpus*
            - paragraph_span_<type>                         : *span location in gold output of the paragraphs
                                                                from corpus*
            NOTE : paragraph_id[k] corresponds to paragraph_span[k]*
            <type> in [
                    w_heading_all_passage,                  # with heading and entire section
                    w/o_heading_all_passage,                # without heading and entire section
                    w_heading_first_sentence_by_paragraph,  # with heading and first sentence of each paragraph
                    w/o_heading_first_sentence_by_paragraph,  # without heading and first sentence of each paragraph


    """
    df = []
    with open(path_file, 'rb') as file:
        for page in tqdm(list(iter_pages(file))):
            if page.child_sections:
                para_ids, para_text = get_paragraphs(page).values()
                for section in page.child_sections:
                    if section.child_sections:
                        texts = get_intro_section(section, tokenizer)
                        texts["query"] = page.page_name + "/" + section.heading
                        texts["outline"] = ["/".join([page.page_name] + [str(s.heading)
                                                                         for s in sectionpath[sectionpath.index(section)
                                                                                              + 1:len(sectionpath)]])
                                            for sectionpath in page.flat_headings_list() if section in sectionpath]
                        texts = get_text(section, texts, tokenizer)
                        texts = get_paragraphs_id_and_span(texts, tokenizer, para_ids, para_text)
                        df.append(texts)

    df = pd.DataFrame(df)
    df = df[["query", "outline",
             "text_w_heading_all_passage", "paragraph_span_w_heading_all_passage",
             "paragraph_id_w_heading_all_passage",
             "text_w/o_heading_all_passage", "paragraph_span_w/o_heading_all_passage",
             "paragraph_id_w/o_heading_all_passage",
             "text_w_heading_first_sentence_by_paragraph", "paragraph_span_w_heading_first_sentence_by_paragraph",
             "paragraph_id_w_heading_first_sentence_by_paragraph",
             "text_w/o_heading_first_sentence_by_paragraph", "paragraph_span_w/o_heading_first_sentence_by_paragraph",
             "paragraph_id_w/o_heading_first_sentence_by_paragraph"]]

    return df


def main(path_file: str,
         path_output_folder: str):
    """
    :param path_file: path to the *.cboor file to process
    :param path_output_folder: path to the save location. In addition, a subfolder for the fold will be created.
    :return:
    """
    t_start = time.time()
    logging.info(f"Start processing {path_file}.")

    logging.info("Parse path_file for 'data type_train_test' (and 'fold' if train data).")
    type_train_test = re.findall(r"train|test", path_file)[0]
    logging.info(f"\tfile type : {type_train_test}.")
    paragraphs_only = True if re.findall(r"paragraphs", path_file) else False
    logging.info(f"\tfile contains {'corpus' if paragraphs_only else 'pages'}.")

    if type_train_test == "train":
        try:
            fold = int(re.findall(r"fold-\d", path_file)[0].replace("fold-", ''))
            path_output_folder = os.path.join(path_output_folder, f"fold-{fold}")
        except IndexError:
            path_output_folder = os.path.join(path_output_folder, f"all-")

    else:  # test
        path_output_folder = os.path.join(path_output_folder, f"test")

    logging.info('Check if output path exists or create it.')
    if not os.path.exists(path_output_folder):
        incremented_path = ""
        for path in path_output_folder.split("/"):
            incremented_path = os.path.join(incremented_path, path)
            try:
                os.mkdir(incremented_path)
            except FileExistsError:
                pass

    if paragraphs_only:  # the file is just a corpus
        logging.info("construct corpus dataset")
        df_corpus = construct_corpus(path_file)
        logging.info("save dataset")
        df_corpus.to_csv(os.path.join(path_output_folder, f"corpus_{type_train_test}.csv"), index=False)
    else:
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

        logging.info("construct article dataset")
        df_article = construct_article(path_file, tokenizer)
        logging.info("\tsave dataset")
        df_article.to_csv(os.path.join(path_output_folder, f"articles_{type_train_test}.csv"), index=False)

        logging.info("construct section dataset")
        df_section = construct_section(path_file, tokenizer)
        logging.info("\tsave dataset")
        df_section.to_csv(os.path.join(path_output_folder, f"sections_{type_train_test}.csv"), index=False)

    logging.info(f"DONE. Elepased time : {time.time() - t_start:.2f}s.\n\n")


if __name__ == "__main__":
    # train
    for i in range(0, 5):
        main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor",
             path_output_folder="../../data_pre_processed")
        main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor-paragraphs.cbor",
             path_output_folder="../../data_pre_processed")
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/train.pages.cbor-paragraphs.cbor",
         path_output_folder="../../data_pre_processed")
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/train.pages.cbor-paragraphs.cbor",
         path_output_folder="../../data_pre_processed")
    # test
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor",
         path_output_folder="../../data_pre_processed")
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor-paragraphs.cbor",
         path_output_folder="../../data_pre_processed")
