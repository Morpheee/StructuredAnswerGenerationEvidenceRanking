#! /usr/bin/env python3
import re
import nltk.data
import pandas as pd
import os
import sys
import logging
import time
sys.path.append("../../.")
from trec_car_tools.python3.trec_car.read_data import iter_pages, iter_paragraphs

logging.basicConfig(level=logging.INFO)

def clean_df(df) :
    for col in df.columns:
        if "text" in col:
            df[col] = df[col].apply(clean_text)
    return df


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
    :param tokenizer: tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :return: df : DataFrame with the column :
        - text : *text from the corpus*
    """
    text = []
    with open(path_file, "rb") as file:
        for paragraph in iter_paragraphs(file):
            if re.match(r"\w+", paragraph.get_text()):
                text.append(str(paragraph))
    df = pd.DataFrame({"text": text})
    return df


def construct_article(path_file: str, tokenizer):
    """
    :param path_file: path to the *.cboor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_w_heading_first_sentence_by_section   : *gold_output with heading and the first sentence of
                                                                each section or subsection*
            - text_w_heading_first_sentence_by_paragraph    : *gold_output with heading and the first sentence of
                                                                each paragraph*
            - text_w_heading_all_passage                    : *gold_output with heading and the all passage*
            - text_w/o_heading_first_sentence_by_section : *gold_output without heading and the first sentence of
                                                                each section or subsection*
            - text_w/o_heading_first_sentence_by_paragraph  : *gold_output without heading and the first sentence of
                                                                each paragraph*
            - text_w/o_heading_all_passage                  : *gold_output without heading and the all passage*
    """
    df = []
    with open(path_file, 'rb') as file:
        for page in iter_pages(file):
            if page.child_sections:
                data_page = dict()
                first_section = page.child_sections[0].get_text_with_headings(False)

                # handle intro
                intro_all_passage_list = []
                intro_first_sentence_list = []
                for passage_skeleton in page.skeleton:
                    passage = passage_skeleton.get_text_with_headings(False)
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
                intro_first_sentence_by_section = intro_first_sentence_list[0]
                data_page["query"] = page.page_name

                # handle rest of text
                data_page["text_w_heading_first_sentence_by_section"] = intro_first_sentence_by_section
                data_page["text_w_heading_first_sentence_by_paragraph"] = intro_first_sentence_by_paragraph
                data_page["text_w_heading_all_passage"] = intro_all_passage
                data_page["text_w/o_heading_first_sentence_by_section"] = intro_first_sentence_by_section
                data_page["text_w/o_heading_first_sentence_by_paragraph"] = intro_first_sentence_by_paragraph
                data_page["text_w/o_heading_all_passage"] = intro_all_passage

                if len(page.outline()) > 0:
                    outline = ["/".join([str(section.heading)
                                         for section in section_path])
                               for section_path in page.flat_headings_list()]

                    # process text with all passage and heading
                    # in addition, get the entire page (but the intro)
                    entire_page = ""
                    for section in page.outline():
                        data_page["text_w_heading_all_passage"] += str(section)
                        entire_page += str(section)
                    # process text with all passage and no heading
                    data_page["text_w/o_heading_all_passage"] += page.get_text()
                    # process text with first sentence by paragraph and no heading
                    for paragraph in page.get_text().split("\n"):
                        if not re.match(r"\w+", paragraph):
                            continue
                        data_page["text_w/o_heading_first_sentence_by_paragraph"] += str(
                            "\n" + tokenizer.tokenize(paragraph)[0])

                    # get all headings
                    headings = []
                    for section_path in page.flat_headings_list():
                        heading = str(len(list(section_path)) * "=" + " " + section_path[-1].heading + " " + len(
                            list(section_path)) * "=")
                        headings.append(heading)
                    for i in range(len(headings)):
                        # extract corresponding section to a given heading
                        heading = headings[i]
                        if i < len(headings) - 1:
                            heading_next = headings[i + 1]
                            page_split = entire_page.split(heading)
                            relevant_split = page_split[1]
                            section = relevant_split.split(heading_next)[0]
                        else:
                            section = entire_page.split(heading)[-1]
                        # get first sentence of section and of each paragraph
                        section_first_sentence_by_paragraph = ""
                        if re.search(r"\w", section):
                            section_first_sentence = tokenizer.tokenize(section)[0]
                            for sentence in section.split("\n"):
                                if not re.match(r"\w+", sentence):
                                    continue
                                section_first_sentence_by_paragraph += str("\n" + tokenizer.tokenize(sentence)[0])
                        else:
                            section_first_sentence = ""

                        # add to corresponding text data
                        data_page["text_w_heading_first_sentence_by_section"] += str(
                            "\n" + heading + "\n" + section_first_sentence)
                        data_page["text_w_heading_first_sentence_by_paragraph"] += str(
                            "\n" + heading + "\n" + section_first_sentence_by_paragraph)
                        data_page["text_w/o_heading_first_sentence_by_section"] += str("\n" + section_first_sentence)
                else:
                    outline = []
                data_page["outline"] = outline
                df.append(data_page)
            else:
                print("not used :", page.page_type)
    df = pd.DataFrame(df)
    return df


def construct_section(path_file: str, tokenizer):
    """
    :param path_file: path to the *.cboor file to process
    :param tokenizer: tokenizer that, when used as tokenizer.tokenize(text) -> output a list of sentences.
    :return: df : DataFrame with the following columns :
            - query                                         : *page_title/section_title*
            - outline                                       : *list of sections and subsections titles*
            - text_w_heading_first_sentence_by_subsection   : *gold_output with heading and the first sentence of
                                                                each subsection*
            - text_w_heading_first_sentence_by_paragraph    : *gold_output with heading and the first sentence of
                                                                each paragraph*
            - text_w_heading_all_passage                    : *gold_output with heading and the all passage*
            - text_w/o_heading_first_sentence_by_subsection : *gold_output without heading and the first sentence of
                                                                each subsection*
            - text_w/o_heading_first_sentence_by_paragraph  : *gold_output without heading and the first sentence of
                                                                each paragraph*
            - text_w/o_heading_all_passage                  : *gold_output without heading and the all passage*
    """
    df = []
    with open(path_file, 'rb') as file:
        for page in iter_pages(file):
            if len(page.outline()) > 0:
                for section in page.child_sections:
                    if section:
                        data_section = dict()
                        # get query (i.e. section title)
                        data_section["query"] = page.page_name + "/" + section.heading
                        # get outline (i.e. subsections titles)
                        outline = ["/".join([str(s.heading)
                                             for s in sectionpath[sectionpath.index(section) + 1:len(sectionpath)]])
                                   for sectionpath in page.flat_headings_list() if section in sectionpath]
                        while '' in outline:
                            outline.remove('')
                        data_section["outline"] = outline
                        data_section["text_w_heading_first_sentence_by_subsection"] = ""
                        data_section["text_w_heading_first_sentence_by_paragraph"] = ""
                        data_section["text_w_heading_all_passage"] = str(section).replace(f"= {section.heading} =", '')
                        data_section["text_w/o_heading_first_sentence_by_subsection"] = ""
                        data_section["text_w/o_heading_first_sentence_by_paragraph"] = ""
                        data_section["text_w/o_heading_all_passage"] = section.get_text()
                        # get all headings
                        headings = re.findall(r"=+(?:\w|\s|\d)+=+", data_section["text_w_heading_all_passage"])
                        # get entire section
                        entire_section = str(section)
                        if not headings:
                            heading = ""
                            try:
                                subsection_first_sentence = tokenizer.tokenize(section.get_text())[0]
                            except:
                                logging.warning(f" > > > Error with page '{page.page_name}', "
                                                f"section '{section.heading}' is empty.")
                                continue
                            subsection_first_sentence_by_paragraph = ""
                            if re.search(r"\w+", section.get_text()):
                                subsection_first_sentence = tokenizer.tokenize(section.get_text())[0]
                                for paragraph in section.get_text().split("\n"):
                                    if not re.match(r"\w+", paragraph):
                                        continue
                                    subsection_first_sentence_by_paragraph += str(
                                        "\n" + tokenizer.tokenize(paragraph)[0])
                            # add to corresponding text data
                            data_section["text_w_heading_first_sentence_by_subsection"] += str(
                                "\n" + heading + "\n" + subsection_first_sentence)
                            data_section["text_w_heading_first_sentence_by_paragraph"] += str(
                                "\n" + heading + "\n" + subsection_first_sentence_by_paragraph)
                            data_section["text_w/o_heading_first_sentence_by_subsection"] += str(
                                "\n" + subsection_first_sentence)
                            data_section["text_w/o_heading_first_sentence_by_paragraph"] += str(
                                "\n" + subsection_first_sentence_by_paragraph)

                        for i in range(len(headings)):
                            heading = headings[i]
                            if i < len(headings) - 1:
                                heading_next = headings[i + 1]
                                section_split = entire_section.split(heading)
                                relevant_split = section_split[1]
                                subsection = relevant_split.split(heading_next)[0]
                            else:
                                subsection = entire_section.split(heading)[1]
                            # get first sentence of section and of each paragraph
                            subsection_first_sentence_by_paragraph = ""
                            if re.search(r"\w+", subsection):
                                subsection_first_sentence = tokenizer.tokenize(subsection)[0]
                                for paragraph in subsection.split("\n"):
                                    if not re.match(r"\w+", paragraph):
                                        continue
                                    subsection_first_sentence_by_paragraph += str(
                                        "\n" + tokenizer.tokenize(paragraph)[0])
                            else:
                                subsection_first_sentence = ""
                            # add to corresponding text data
                            data_section["text_w_heading_first_sentence_by_subsection"] += str(
                                "\n" + heading + "\n" + subsection_first_sentence)
                            data_section["text_w_heading_first_sentence_by_paragraph"] += str(
                                "\n" + heading + "\n" + subsection_first_sentence_by_paragraph)
                            data_section["text_w/o_heading_first_sentence_by_subsection"] += str(
                                "\n" + subsection_first_sentence)
                            data_section["text_w/o_heading_first_sentence_by_paragraph"] += str(
                                "\n" + subsection_first_sentence_by_paragraph)
                        df.append(data_section)
                    else:
                        print("no used :", section)
    df = pd.DataFrame(df)
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
    logging.info(f"file type : {type_train_test}.")
    paragraphs_only = True if re.findall("paragraphs", path_file) else False
    logging.info(f"file is {'a corpus' if paragraphs_only else 'entire pages'}.")

    if type_train_test == "train":
        fold = int(re.findall(r"fold-\d", path_file)[0].replace("fold-", ''))
        path_output_folder = os.path.join(path_output_folder, f"fold-{fold}")
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
        logging.info("clean text features.")
        df_corpus = clean_df(df_corpus)
        logging.info("save dataset")
        df_corpus.to_csv(os.path.join(path_output_folder, f"corpus_{type_train_test}.csv"), index=False)
    else:
        logging.info("Get tokenizer.")
        try:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt')
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        logging.info("construct article dataset")
        df_article = construct_article(path_file, tokenizer)
        logging.info("\tclean text features")
        df_article = clean_df(df_article)
        logging.info("\tsave dataset")
        df_article.to_csv(os.path.join(path_output_folder, f"articles_{type_train_test}.csv"), index=False)

        logging.info("construct section dataset")
        df_section = construct_section(path_file, tokenizer)
        logging.info("\tclean text features")
        df_section = clean_df(df_section)
        logging.info("\tsave dataset")
        df_section.to_csv(os.path.join(path_output_folder, f"sections_{type_train_test}.csv"), index=False)

    logging.info(f"DONE. Elepased time : {time.time() - t_start:.2f}s.\n\n")


if __name__ == "__main__":
    # train
    # for i in range(1, 5):
    #     main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor",
    #          path_output_folder="../../data_pre_processed")
    #     main(path_file=f"../../Data/benchmarkY1/benchmarkY1-train/fold-{i}-train.pages.cbor-paragraphs.cbor",
    #          path_output_folder="../../data_pre_processed")
    # test
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor",
         path_output_folder="../../data_pre_processed")
    main(path_file=f"../../Data/benchmarkY1/benchmarkY1-test/test.pages.cbor-paragraphs.cbor",
         path_output_folder="../../data_pre_processed")
