##### triples/asoif.1912271956.ttl
标识出实体的类型：character、castle、house、secondary_entity
##### triples/asoif.1912280836.ttl
在triples/asoif.1912271956.ttl的基础上。
去掉secondary entity
##### triples/asoif.1912280919.ttl
在asoif.1912280836.ttl的基础上。
去掉原著書目\原著书目、第*季、出场集数、出场季数、提及集数
- character: 2,240
- castle: 171
- house: 461
- all: 15,487
##### triples/asoif.1912301025.ttl
在asoif.1912280919.ttl基础上，增加名和姓
- all: 19,216

##### triples/asoif.1912301959.ttl
加入培提尔·贝里席
##### triples/asoif.1912302006.ttl
在asoif.1912301959.ttl的基础上。
去掉原著書目\原著书目、第*季、出场集数、出场季数、提及集数
##### triples/asoif.1912302025.ttl
在triples/asoif.1912302006.ttl的基础上。
去掉secondary entity
- character: 2,241
- castle: 171
- house: 461
- all: 15,499
##### triples/asoif.1912302030.ttl
在asoif.1912280919.ttl基础上，增加名和姓
- all: 19,230
##### triples/asoif.1912310949.ttl
在triples/asoif.1912302030.ttl基础上，
将`第.任.{{0,1}}<人名>`替换为实体
##### triples/asoif.1912310954.ttl
在triples/asoif.1912310949.ttl基础上，
将`r:现任领主`的tail替换为实体

##### triples/asoif.1912311025.ttl
在triples/asoif.1912310954.ttl基础上，
替换了一些实体，主要是`继承|继任|父亲|母亲|子嗣|兄弟姐妹`这些关系的tail实体

##### triples/asoif.1912310949-s.ttl
在triples/asoif.1912310949.ttl的基础上，将繁体字转换为简体字

##### triples/asoif.1912311434.ttl
在triples/asoif.1912310949-s.ttl的基础上，
- `r:王后` -> `r:配偶`
- `r:继承者` -> `r:继承人`
> addition: `e:乔佛里·拜拉席恩	r:全称	"拜拉席恩家族和兰尼斯特家族的乔佛里一世".`

##### triples/asoif.1912311839.ttl
在triples/asoif.1912311434.ttl的基础上，
- `r:丈夫` -> `r:配偶`

##### TODO (Tail matching):
现在图谱中有很多三元组的tail实体为literal的，literal的内容可能指向一个实体
比如`继承|继任|父亲|母亲|子嗣|兄弟姐妹|现任领主`这些关系的tail按理说应该指向一个实体，但是很多都是literal的
```
e:史蒂芬·拜拉席恩 r:子嗣 "劳勃"
e:史蒂芬·拜拉席恩 r:子嗣 "史坦尼斯"
e:史蒂芬·拜拉席恩 r:子嗣 "蓝礼"
```
这里面的"劳勃"等应该替换为`e:劳勃·拜拉席恩`。但问题在于这个替换无法使用简单的匹配来进行。因为"劳勃"这个词也可以匹配到其他名字为"劳勃"的人物（如`e:劳勃·艾林`）
因此考虑借助文本内容进行辅助。例如上面的例子中，e:史蒂芬·拜拉席恩与e:劳勃·拜拉席恩在文本中共现的次数较多，因此可以把这里的"劳勃"替换为`e:劳勃·拜拉席恩`


**具体步骤：**
1. 找出所有关系为`继承|继任|父亲|母亲|子嗣|兄弟姐妹|现任领主`且tail为literal的三元组，记为集合S

2. 文本匹配
    ```python
    for s in S:
       找出所有能和s.tail匹配上的e，记为E_s
       E_s中所有实体计算与s.tail的文本相似度dist(e, s.tail)
    ```
3. 共现度
    ```python
    for s in S:
       E_s中所有实体计算与s.tail计算在文本中的共现度occur(e, s.tail)
    ```
4. 替换
    将`S.tail`替换为相似度和共现度都高的那个实体
    
##### triples/asoif.2001012253.ttl
按上述的方式处理，可能有些替换是错的
具体处理参照processed_data/processed_datd_description.md
处理后总的实体数为6364 -> 6,008 
```sparql
PREFIX	r:	<http://kg.course/action/>
PREFIX	e:	<http://kg.course/entity/>
SELECT DISTINCT ?e
WHERE{
  {?e ?r ?o}  
  UNION{?s ?r ?e}
  MINUS{?s r:name ?e}
  MINUS{?s r:名 ?e}
  MINUS{?s r:姓 ?e}
  MINUS{?s r:type ?e}
#  ?s r:王室 ?o
#  e:拜拉席恩家族 ?r ?o
}
```


##### triples/asoif.2001021922.ttl
发现bug：添加到实体集合时，没有先对实体名称进行清洗，导致有些清洗后和secondary实体重名的实体被删除
##### triples/asoif.2001021935.ttl
在triples/asoif.2001021935.ttl的基础上。
去掉secondary entity。
##### triples/asoif.2001021941.ttl
在asoif.1912301959.ttl的基础上。
去掉原著書目\原著书目、第*季、出场集数、出场季数、提及集数。
增加名和姓。
##### triples/asoif.2001021948.ttl
在asoif.2001021941.ttl的基础上。
去掉一些纯英语的三元组
##### triples/asoif.2001021948-s.ttl
在asoif.2001021948.ttl的基础上。繁体转换为简体
##### triples/asoif.2001021955.ttl
- `r:王后` -> `r:配偶`
- `r:继承者` -> `r:继承人`
- `r:丈夫` -> `r:配偶`
> addition: <br>
>`e:乔佛里·拜拉席恩	r:全称	"拜拉席恩家族和兰尼斯特家族的乔佛里一世".`<br>
> `e:兰尼斯特家族	r:封号	"凯岩王".`<br>
> `e:兰尼斯特家族	r:封号	"凯岩城公爵".`<br>
> `e:兰尼斯特家族	r:封号	"兰尼斯港之盾".`<br>
> `e:兰尼斯特家族	r:封号	"西境守护".`<br>
> `e:兰尼斯特家族	r:封号	"西境统领".`<br>
> `e:兰尼斯特家族	r:创建	"机灵的兰恩创建于英雄纪元".`<br>
>


##### triples/asoif.2001022253.ttl
发现bug：clean text的时候会把中文标点如？！、；，（）《》“”去掉
##### triples/asoif.2001030817.ttl
在triples/asoif.2001022253.ttl的基础上。
去掉secondary entity。
##### triples/asoif.2001030820.ttl
在triples/asoif.2001030817.ttl的基础上。
去掉原著書目\原著书目、第*季、出场集数、出场季数、提及集数。
去掉一些纯英语的三元组
> addition: <br>
>`e:乔佛里·拜拉席恩	r:全称	"拜拉席恩家族和兰尼斯特家族的乔佛里一世".`<br>
> `e:兰尼斯特家族	r:封号	"凯岩王".`<br>
> `e:兰尼斯特家族	r:封号	"凯岩城公爵".`<br>
> `e:兰尼斯特家族	r:封号	"兰尼斯港之盾".`<br>
> `e:兰尼斯特家族	r:封号	"西境守护".`<br>
> `e:兰尼斯特家族	r:封号	"西境统领".`<br>
> `e:兰尼斯特家族	r:创建	"机灵的兰恩创建于英雄纪元".`<br>
##### triples/asoif.2001030820-s.ttl
在asoif.2001021948.ttl的基础上。繁体转换为简体

##### triples/asoif.20010030828.ttl
在triples/asoif.2001030820-s.ttl的基础上。
增加名和姓。
##### triples/asoif.2001020836.ttl
在triples/asoif.20010030828.ttl的基础上。
- `r:王后` -> `r:配偶`
- `r:继承者` -> `r:继承人`
- `r:丈夫` -> `r:配偶`

共有2,870个实体, 其中character: 2,260个, castle: 171个, house: 439个

##### triples/asoif.2001041219.ttl
使用 processed_data/candidate_entity_replacement_list_v4.jsonl进行实体替换

##### triples/asoif.2001042058.ttl
使用processed_data/candidate_entity_replacement_list_v6.jsonl进行实体替换