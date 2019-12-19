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
            ret.append((e, "name", e))
        return ret


ER = EntityRecord()


def clean_text(s):
    s = re.sub(r"[^\x00-\xff\u4E00-\u9FA5]", "", s)  # 去掉不是英文也不是中文的奇怪符号
    s = re.sub(r"\s*", "", s)
    return s


def to_file(filename, tuples, is_entity):
    with open(filename, mode="a", encoding="utf-8", errors="ignore", ) as f:
        if is_entity:
            for s, p, o in tuples:
                f.write(f"e:{s}\ta:{p}\t\"{o}\"\n")
        else:
            for s, p, o in tuples:
                f.write(f"e:{s}\ta:{p}\te:{o}\n")


def sep_tag(elems, split_pattern):
    ret = []
    for e in elems:
        e_str = re.split(split_pattern, etree.tostring(e).decode('utf-8'))
        for s in e_str:
            try:
                s = "".join(etree.HTML(s).xpath("//text()")).strip()
                s = clean_text(s)
                if not re.fullmatch(r"\s*", s):
                    ret.append(s)
            except:
                pass
    return ret


def get_page(http, url):
    return http.request('GET', url).data.decode('utf-8')


def get_characters(root_url, page):
    doc = etree.HTML(page)
    urls = doc.xpath("//div[contains(concat(' ', @class, ' '), concat(' ','mw-category',' ')) ]//a/@href")
    for i, u in enumerate(urls):
        urls[i] = root_url + u
    return urls


def get_info(page):
    ret = []
    doc = etree.HTML(page)
    data = doc.xpath(
        "//*[contains(concat(' ', @class, ' '), concat(' ','infobox',' '))]/tr[contains(concat(' ', @class, ' '), concat(' ','',' '))]")
    s = "".join(doc.xpath(".//*[contains(concat(' ', @class, ' '), concat(' ','infobox-title',' '))]//big//text()"))
    ER.entity_set.add(s)
    for i in data:
        p = "".join(
            i.xpath(".//*[contains(concat(' ', @class, ' '), concat(' ','infobox-label',' '))]//text()")).strip()
        o = i.xpath(".//*[contains(concat(' ', @class, ' '), concat(' ','infobox-data',' '))]")
        o = sep_tag(o, r"<br>|<br/>")
        for o_ in o:
            ER.entity_set.add(o_)
            ret.append((clean_text(s), clean_text(p), clean_text(o_)))
    return ret


def get_a_category(category):
    http = urllib3.PoolManager()
    url_page = get_page(http, f"https://asoiaf.huijiwiki.com/wiki/Category:{quote(category)}")
    urls = get_characters("https://asoiaf.huijiwiki.com", url_page)
    kg = []
    for U in urls:
        print(unquote(U))
        page = get_page(http, U)
        res = get_info(page)
        kg.extend(res)
    to_file("冰与火之歌.ttl", kg, False)
    print(f"category {category}, got")


if __name__ == "__main__":
    Category = ["君王", "平民", "贵族", "骑士"]
    for c in Category:
        get_a_category(c)
    to_file("冰与火之歌.ttl", ER.form_entity_tuples(), True)
