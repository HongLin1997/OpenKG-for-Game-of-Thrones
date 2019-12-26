import os
import pickle
import urllib3
from urllib.parse import quote, unquote
import re
from lxml.html import etree


class EntityRecord:
    def __init__(self):
        self.entity_set = set()

    def form_entity_tuples(self):
        ret = []
        for e in self.entity_set:
            ret.append((f"e:{e}", "r:name", f"\"{e}\""))
        return ret


ER = EntityRecord()
INCRE_PATH = "triples/incremental files"

def clean_text(s):
    s = re.sub(r"[^\x00-\xff\u4E00-\u9FA5]", "", s)  # 去掉不是英文也不是中文的奇怪符号
    s = re.sub(r"\(", "（", s)  # 去掉不可用符号
    s = re.sub(r"\)", "）", s)  # 去掉不可用符号
    s = re.sub(r"\"", "”", s)  # 去掉不可用符号
    s = re.sub(r"\\", "、", s)  # 去掉不可用符号
    s = re.sub(r"\+/-", "约", s)  # 去掉不可用符号
    s = re.sub(r"-", "至", s)  # 去掉不可用符号
    s = re.sub(r"\[\d+\]", "", s)  # 去掉不可用符号
    s = re.sub(r"\s*", "", s)
    return s


def sep_tag(elems, split_pattern):
    """
    在html元素的字符串中进行切割
    :param elems: html元素
    :param split_pattern: 切割所使用的正则表达式
    :return: list of str, 每个字符串都经过清洗
    """
    ret = []
    e_str = re.split(split_pattern, etree.tostring(elems).decode('utf-8'))
    for s in e_str:
        try:
            s = "".join(etree.HTML(s).xpath("//text()")).strip()
            s = clean_text(s)
            if not re.fullmatch(r"\s*", s):
                ret.append(s)
        except:
            pass
    return ret


def dict2file(dict_: dict, filename: str, filetype: int, **kwargs):
    """
    将字典数据写入文件
    :param dict_:
    :param filename:
    :param filetype:
        0: 一行为一项的文本文件
        1: pickle
    :return:
    """
    with open(filename, **kwargs) as f:
        if filetype == 0:
            for key, value in dict_.items():
                f.write(f"{key}: {value}\n")
        elif filetype == 1:
            pickle.dump(dict_, f)


def triple2file(triples, filename, **kwargs):
    with open(filename, **kwargs) as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}.\n")


def apply_xpath2url(url: str, xpath: str):
    """
    返回element
    :param url:
    :param xpath:
    :return:
    """
    http = urllib3.PoolManager()
    page = http.request('GET', url).data.decode('utf-8')
    doc = etree.HTML(page)
    return doc.xpath(xpath)


def apply_xpath2element(elem, xpath: str):
    return elem.xpath(xpath)


def extract_text(elem):
    text = "".join(elem.xpath(".//text()")).strip()
    return text


def get_index_pages(begin_url: str, n_of_page: int, category: str):
    """
    从begin_url开始获取页面索引
    :param category:
    :param n_of_page:
    :param begin_url:
    :return: dict, key=名称, value=url
    """
    ret = dict()
    url_head = "https://asoiaf.huijiwiki.com"
    url = begin_url
    xpath = ".//div[contains(concat(' ', @class, ' '), concat(' ','mw-category-generated',' '))]" \
            "//div[contains(concat(' ', @id, ' '), concat(' ','mw-pages',' '))]"
    for i in range(n_of_page):
        print(f"{i}-th page, get {category} from {unquote(url)}")
        mv_pages = apply_xpath2url(url, xpath)[0]
        names = apply_xpath2element(mv_pages, ".//li//text()")
        urls = apply_xpath2element(mv_pages, ".//li//a/@href")
        for n, u in zip(names, urls):
            print(f"name: {n}, url: {unquote(u)}")
            ret[n] = url_head + u
        try:
            url = url_head + apply_xpath2element(mv_pages, "./a[position()=2]//@href")[0]
        except BaseException as e:
            print(e)
            break
    print(f"finished, all {len(ret)} {category}")
    return ret


def get_info(name: str, url: str):
    ret = []
    header_xpath = "//*[contains(concat(' ', @id, ' '), concat(' ','firstHeading',' '))]//h1//text()"
    header = "".join(apply_xpath2url(url, header_xpath)).strip()
    if header != name:
        with open(os.path.join(INCRE_PATH, "name_mismatch.log"), mode="a", encoding="utf-8", errors="ignore") as f:
            f.write(f"{name}--{header}: {url}\n")
    xpath = "//div[contains(concat(' ', @id, ' '), concat(' ','mw-content-text',' '))]" \
            "//*[contains(@class, 'infobox')]"
    # "/tr[contains(concat(' ', @class, ' '), concat(' ','',' '))]"
    data = apply_xpath2url(url, xpath)
    ER.entity_set.add(name)
    for i in data:
        # 获取一个属性
        infobox_labels = i.xpath(".//*[contains(@class, 'infobox-label')]")
        infobox_data = i.xpath(".//*[contains(@class, 'infobox-data')]")
        for p, o in zip(infobox_labels, infobox_data):
            p = "".join(p.xpath(".//text()"))
            o = sep_tag(o, r"<br>|<br/>")
            # 一个属性有多个属性值
            for o_ in o:
                ER.entity_set.add(o_)
                ret.append((f"e:{clean_text(name)}", f"r:{clean_text(p)}", f"e:{clean_text(o_)}"))
    return ret


def get_index_main():
    res = dict()
    character_index = get_index_pages("https://asoiaf.huijiwiki.com/wiki/Category:%E4%BA%BA%E7%89%A9",
                                      15, "character")
    res.update(character_index)
    house_index = get_index_pages("https://asoiaf.huijiwiki.com/wiki/Category:%E8%B4%B5%E6%97%8F%E5%AE%B6%E6%97%8F",
                                  5, "house")
    res.update(house_index)
    castle_index = get_index_pages("https://asoiaf.huijiwiki.com/wiki/Category:%E5%9F%8E%E5%A0%A1",
                                   2, "castle")
    res.update(castle_index)
    dict2file(res, "triples/index", 0, mode="w", encoding="utf-8", errors="ignore")
    dict2file(res, "triples/index.pkl", 1, mode="wb")


def get__info_main():
    with open("triples/index.pkl", mode="rb") as f:
        name2url = pickle.load(f)
        for n, u in name2url.items():
            print(f"get {n}'s info")
            kg = get_info(n, u)
            if len(kg) < 1:
                with open(os.path.join(INCRE_PATH, "no_triples.log"), mode="a", encoding="utf-8", errors="ignore") as f:
                    f.write(f"{n}: {u}\n")
            triple2file(kg, os.path.join(INCRE_PATH, "asoif.ttl"), mode="a", encoding="utf-8", errors="ignore")
        entities_kg = ER.form_entity_tuples()
        triple2file(entities_kg, os.path.join(INCRE_PATH, "asoif.ttl"), mode="a", encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    # get_index_main()
    get__info_main()
