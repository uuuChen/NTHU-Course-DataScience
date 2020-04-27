from bs4 import BeautifulSoup
import requests
import numpy as np

INPUT_FILE_PATH = './input_hw1.txt'
OUTPUT_FILE_PATH = './108062566_hw1_output.txt'
PARTS_OF_URL = ["https://www.blockchain.com/eth/address/", "?view=standard"]
PERSON_ATTR_KEYS = ['from_hash', 'Nonce', 'Number of Transactions', 'Final Balance', 'Total Sent', 'Total Received',
                    'Total Fees']
ITER_STEPS = 4
SPLIT_LINE = "-" * 74 + "\n"

ATTR_VALUE_CLASS_STR = "sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk"
HASH_CLASS_STR = "sc-1r996ns-0 dEMBMB sc-1tbyx6t-1 gzmhhS iklhnl-0 dVkJbV"
BLOCK_CLASS_STR = "sc-1fp9csv-0 gkLWFf"
TRANSFER_CLASS_STR = "sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk sc-85fclk-0 gskKpd"


def conv_hash2url(hash_str):
    return PARTS_OF_URL[0] + hash_str + PARTS_OF_URL[1]


def get_input_urls(file_path):
    urls = list()
    with open(file_path, 'r') as fp:
        for hash in fp.readlines():
            urls.append(conv_hash2url(hash.strip()))  # convert hash to url
    return urls


def get_person_attrs(soup, tag_name, class_name):
    attrs = dict()
    num_of_attrs = len(PERSON_ATTR_KEYS)
    tags = soup.find_all(tag_name, class_name)
    texts = np.array([tag.get_text() for tag in tags])[:num_of_attrs]
    for i, key in list(enumerate(PERSON_ATTR_KEYS)):  # *(i, key): [(0, 'from_hash'), (1, 'Nonce'), ...]
        attrs[key] = texts[i]
    return attrs


def get_last_trans_attrs(soup, tag_name, class_name):
    attrs = dict()
    attrs['Date'] = attrs['To'] = attrs['Amount'] = None
    blocks = soup.find_all(tag_name, class_name)
    for block in reversed(blocks):  # want to get the oldest transfer hash, so reverse the order of the blocks
        amount_tag = block.find("span", TRANSFER_CLASS_STR)
        if amount_tag is not None:  # the type of the amount is "transfer", so break and return
            attrs['Date'] = block.find("span", ATTR_VALUE_CLASS_STR).get_text()
            attrs['To'] = block.find_all("a", HASH_CLASS_STR)[-1].get_text()  # "to_hash" is the last element of the array
            attrs['Amount'] = amount_tag.get_text()
            break
    return attrs


def log_output(file_path, log_str):
    with open(file_path, "a", encoding="utf-8") as fp:
        fp.write(log_str)


def conv_results2str(show_attrs):
    results_str = str()
    for key in show_attrs:
        value = show_attrs[key]
        if value:  # only when value is not None, it needs to be shown
            results_str += '{}: {}\n'.format(key, value)
    results_str += SPLIT_LINE
    return results_str


def web_crawler(urls, iter_steps, output_file_path):
    for counts, url in list(enumerate(urls, start=1)):
        print('url-counts: {}'.format(counts))
        hashes_log = list()
        next_page_url = url
        for step in range(1, iter_steps+1):
            print('step {}'.format(step))
            r = requests.get(next_page_url)
            html_str = r.text
            soup = BeautifulSoup(html_str, features="html.parser")

            person_attrs = get_person_attrs(soup, "span", ATTR_VALUE_CLASS_STR)
            from_hash = person_attrs['from_hash']
            hashes_log.append(from_hash)
            del person_attrs['from_hash']  # "from_hash" is not used in output.txt, so delete it

            last_trans_attrs = get_last_trans_attrs(soup, "div", BLOCK_CLASS_STR)
            to_hash = last_trans_attrs['To']
            results_str = conv_results2str({**person_attrs, **last_trans_attrs})  # merge two dict and pass to the function
            log_output(output_file_path, results_str)

            if to_hash is None:  # if true means no next, break and log to output_hw1.txt
                break
            next_page_url = conv_hash2url(to_hash)

        hashes_str = ' -> '.join(hashes_log) + "\n" + SPLIT_LINE
        log_output(output_file_path, hashes_str)


def main():
    urls = get_input_urls(INPUT_FILE_PATH)
    web_crawler(urls, ITER_STEPS, OUTPUT_FILE_PATH)


if __name__ == '__main__':
    main()





