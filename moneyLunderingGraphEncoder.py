import pandas as pd
import numpy as np
import json
import networkx as nx
import argparse

# 将此表替换
# graphFile = "新建 XLSX 工作表.xlsx"
# cityFile = "area-china-json\\city.json"

# cityList = []
# with open(cityFile, "rb") as f:
#     for item in f.readlines():
#         cityJson = json.loads(item)
#         # city = json.dumps(cityJson, ensure_ascii=False)
#         cityList.append(cityJson["province_name"] + cityJson["name"])
# print(cityList)

# http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2018/
# 2018年统计用区划代码和城乡划分代码(截止2018年10月31日),仅统计到市一级
cityList = ['北京市市辖区', '天津市市辖区', '河北省石家庄市', '河北省唐山市', '河北省秦皇岛市', '河北省邯郸市', '河北省邢台市', '河北省保定市', '河北省张家口市', '河北省承德市', '河北省沧州市', '河北省廊坊市', '河北省衡水市', '山西省太原市', '山西省大同市', '山西省阳泉市', '山西省长治市', '山西省晋城市', '山西省朔州市', '山西省晋中市', '山西省运城市', '山西省忻州市', '山西省临汾市', '山西省吕梁市', '内蒙古自治区呼和浩特市', '内蒙古自治区包头市', '内蒙古自治区乌海市', '内蒙古自治区赤峰市', '内蒙古自治区通辽市', '内蒙古自治区鄂尔多斯市', '内蒙古自治区呼伦贝尔市', '内蒙古自治区巴彦淖尔市', '内蒙古自治区乌兰察布市', '内蒙古自治区兴安盟', '内蒙古自治区锡林郭勒盟', '内蒙古自治区阿拉善盟', '辽宁省沈阳市', '辽宁省大连市', '辽宁省鞍山市', '辽宁省抚顺市', '辽宁省本溪市', '辽宁省丹东市', '辽宁省锦州市', '辽宁省营口市', '辽宁省阜新市', '辽宁省辽阳市', '辽宁省盘锦市', '辽宁省铁岭市', '辽宁省朝阳市', '辽宁省葫芦岛市', '吉林省长春市', '吉林省吉林市', '吉林省四平市', '吉林省辽源市', '吉林省通化市', '吉林省白山市', '吉林省松原市', '吉林省白城市', '吉林省延边朝鲜族自治州', '黑龙江省哈尔滨市', '黑龙江省齐齐哈尔市', '黑龙江省鸡西市', '黑龙江省鹤岗市', '黑龙江省双鸭山市', '黑龙江省大庆市', '黑龙江省伊春市', '黑龙江省佳木斯市', '黑龙江省七台河市', '黑龙江省牡丹江市', '黑龙江省黑河市', '黑龙江省绥化市', '黑龙江省大兴安岭地区', '上海市市辖区', '江苏省南京市', '江苏省无锡市', '江苏省徐州市', '江苏省常州市', '江苏省苏州市', '江苏省南通市', '江苏省连云港市', '江苏省淮安市', '江苏省盐城市', '江苏省扬州市', '江苏省镇江市', '江苏省泰州市', '江苏省宿迁市', '浙江省杭州市', '浙江省宁波市', '浙江省温州市', '浙江省嘉兴市', '浙江省湖州市', '浙江省绍兴市', '浙江省金华市', '浙江省衢州市', '浙江省舟山市', '浙江省台州市', '浙江省丽水市', '安徽省合肥市', '安徽省芜湖市', '安徽省蚌埠市', '安徽省淮南市', '安徽省马鞍山市', '安徽省淮北市', '安徽省铜陵市', '安徽省安庆市', '安徽省黄山市', '安徽省滁州市', '安徽省阜阳市', '安徽省宿州市', '安徽省六安市', '安徽省亳州市', '安徽省池州市', '安徽省宣城市', '福建省福州市', '福建省厦门市', '福建省莆田市', '福建省三明市', '福建省泉州市', '福建省漳州市', '福建省南平市', '福建省龙岩市', '福建省宁德市', '江西省南昌市', '江西省景德镇市', '江西省萍乡市', '江西省九江市', '江西省新余市', '江西省鹰潭市', '江西省赣州市', '江西省吉安市', '江西省宜春市', '江西省抚州市', '江西省上饶市', '山东省济南市', '山东省青岛市', '山东省淄博市', '山东省枣庄市', '山东省东营市', '山东省烟台市', '山东省潍坊市', '山东省济宁市', '山东省泰安市', '山东省威海市', '山东省日照市', '山东省莱芜市', '山东省临沂市', '山东省德州市', '山东省聊城市', '山东省滨州市', '山东省菏泽市', '河南省郑州市', '河南省开封市', '河南省洛阳市', '河南省平顶山市', '河南省安阳市', '河南省鹤壁市', '河南省新乡市', '河南省焦作市', '河南省濮阳市', '河南省许昌市', '河南省漯河市', '河南省三门峡市', '河南省南阳市', '河南省商丘市', '河南省信阳市', '河南省周口市', '河南省驻马店市', '河南省省直辖县级行政区划', '湖北省武汉市', '湖北省黄石市', '湖北省十堰市', '湖北省宜昌市', '湖北省襄阳市', '湖北省鄂州市', '湖北省荆门市', '湖北省孝感市', '湖北省荆州市', '湖北省黄冈市', '湖北省咸宁市', '湖北省随州市', '湖北省恩施土家族苗族自治州', '湖北省省直辖县级行政区划', '湖南省长沙市', '湖南省株洲市', '湖南省湘潭市', '湖南省衡阳市', '湖南省邵阳市', '湖南省岳阳市', '湖南省常德市', '湖南省张家界市', '湖南省益阳市', '湖南省郴州市', '湖南省永州市', '湖南省怀化市', '湖南省娄底市', '湖南省湘西土家族苗族自治州', '广东省广州市', '广东省韶关市', '广东省深圳市', '广东省珠海市', '广东省汕头市', '广东省佛山市', '广东省江门市', '广东省湛江市', '广东省茂名市', '广东省肇庆市', '广东省惠州市', '广东省梅州市', '广东省汕尾市', '广东省河源市', '广东省阳江市', '广东省清远市', '广东省东莞市', '广东省中山市', '广东省潮州市', '广东省揭阳市', '广东省云浮市', '广西壮族自治区南宁市', '广西壮族自治区柳州市', '广西壮族自治区桂林市', '广西壮族自治区梧州市', '广西壮族自治区北海市', '广西壮族自治区防城港市', '广西壮族自治区钦州市', '广西壮族自治区贵港市', '广西壮族自治区玉林市', '广西壮族自治区百色市', '广西壮族自治区贺州市', '广西壮族自治区河池市', '广西壮族自治区来宾市', '广西壮族自治区崇左市', '海南省海口市', '海南省三亚市', '海南省三沙市', '海南省儋州市', '海南省省直辖县级行政区划', '重庆市市辖区', '重庆市县', '四川省成都市', '四川省自贡市', '四川省攀枝花市', '四川省泸州市', '四川省德阳市', '四川省绵阳市', '四川省广元市', '四川省遂宁市', '四川省内江市', '四川省乐山市', '四川省南充市', '四川省眉山市', '四川省宜宾市', '四川省广安市', '四川省达州市', '四川省雅安市', '四川省巴中市', '四川省资阳市', '四川省阿坝藏族羌族自治州', '四川省甘孜藏族自治州', '四川省凉山彝族自治州', '贵州省贵阳市', '贵州省六盘水市', '贵州省遵义市', '贵州省安顺市', '贵州省毕节市', '贵州省铜仁市', '贵州省黔西南布依族苗族自治州', '贵州省黔东南苗族侗族自治州', '贵州省黔南布依族苗族自治州', '云南省昆明市', '云南省曲靖市', '云南省玉溪市', '云南省保山市', '云南省昭通市', '云南省丽江市', '云南省普洱市', '云南省临沧市', '云南省楚雄彝族自治州', '云南省红河哈尼族彝族自治州', '云南省文山壮族苗族自治州', '云南省西双版纳傣族自治州', '云南省大理白族自治州', '云南省德宏傣族景颇族自治州', '云南省怒江傈僳族自治州', '云南省迪庆藏族自治州', '西藏自治区拉萨市', '西藏自治区日喀则市', '西藏自治区昌都市', '西藏自治区林芝市', '西藏自治区山南市', '西藏自治区那曲市', '西藏自治区阿里地区', '陕西省西安市', '陕西省铜川市', '陕西省宝鸡市', '陕西省咸阳市', '陕西省渭南市', '陕西省延安市', '陕西省汉中市', '陕西省榆林市', '陕西省安康市', '陕西省商洛市', '甘肃省兰州市', '甘肃省嘉峪关市', '甘肃省金昌市', '甘肃省白银市', '甘肃省天水市', '甘肃省武威市', '甘肃省张掖市', '甘肃省平凉市', '甘肃省酒泉市', '甘肃省庆阳市', '甘肃省定西市', '甘肃省陇南市', '甘肃省临夏回族自治州', '甘肃省甘南藏族自治州', '青海省西宁市', '青海省海东市', '青海省海北藏族自治州', '青海省黄南藏族自治州', '青海省海南藏族自治州', '青海省果洛藏族自治州', '青海省玉树藏族自治州', '青海省海西蒙古族藏族自治州', '宁夏回族自治区银川市', '宁夏回族自治区石嘴山市', '宁夏回族自治区吴忠市', '宁夏回族自治区固原市', '宁夏回族自治区中卫市', '新疆维吾尔自治区乌鲁木齐市', '新疆维吾尔自治区克拉玛依市', '新疆维吾尔自治区吐鲁番市', '新疆维吾尔自治区哈密市', '新疆维吾尔自治区昌吉回族自治州', '新疆维吾尔自治区博尔塔拉蒙古自治州', '新疆维吾尔自治区巴音郭楞蒙古自治州', '新疆维吾尔自治区阿克苏地区', '新疆维吾尔自治区克孜勒苏柯尔克孜自治州', '新疆维吾尔自治区喀什地区', '新疆维吾尔自治区和田地区', '新疆维吾尔自治区伊犁哈萨克自治州', '新疆维吾尔自治区塔城地区', '新疆维吾尔自治区阿勒泰地区', '新疆维吾尔自治区自治区直辖县级行政']

def buildMoneyLaunderingGraph(graphFile):
    # step 1: read excel file, and build the one-hot dict for each attr
    G = pd.read_excel(graphFile)

    JIAOYIFANGSHI = G["JIAOYIFANGSHI"].str.split('-', expand=True)
    JIAOYIFANGSHI0 = JIAOYIFANGSHI[0]
    JIAOYIFANGSHI0.drop_duplicates(inplace=True)
    JIAOYIFANGSHI0Array = np.array(JIAOYIFANGSHI0)
    # p = np.where(JIAOYIFANGSHI0Array=="")
    # print(p, JIAOYIFANGSHI0Array.shape[0])

    JIAOYIFANGSHI1 = JIAOYIFANGSHI[1]
    JIAOYIFANGSHI1.drop_duplicates(inplace=True)
    JIAOYIFANGSHI1Array = np.array(JIAOYIFANGSHI1)
    # p = np.where(JIAOYIFANGSHI1Array=="非现钞")
    # print(p, JIAOYIFANGSHI1Array.shape[0])

    JIAOYIFANGSHI2 = JIAOYIFANGSHI[2]
    JIAOYIFANGSHI2.drop_duplicates(inplace=True)
    JIAOYIFANGSHI2Array = np.array(JIAOYIFANGSHI2)
    # p = np.where(JIAOYIFANGSHI2Array=="银行卡")
    # print(p, JIAOYIFANGSHI2Array.shape[0])

    FUKUANFANGYINHANG = G["FUKUANFANGYINHANG"]
    FUKUANFANGYINHANG.drop_duplicates(inplace=True)
    FUKUANFANGYINHANGArray = np.array(FUKUANFANGYINHANG)
    # p = np.where(FUKUANFANGYINHANGArray==FUKUANFANGYINHANGArray)
    # print(p, FUKUANFANGYINHANGArray.shape[0])

    HUOBIMINGCHENG = G["HUOBIMINGCHENG"]
    HUOBIMINGCHENG.drop_duplicates(inplace=True)
    HUOBIMINGCHENGArray = np.array(HUOBIMINGCHENG)
    # p = np.where(HUOBIMINGCHENGArray=="人民币元")
    # print(p)

    JIAYIFASHENGDI = G["JIAYIFASHENGDI"]
    JIAYIFASHENGDIArray = np.array(JIAYIFASHENGDI)

    # step 2: build one-hot vector row by row using one-hot dict generated before
    networkGraph = nx.MultiDiGraph()
    # To unify node ID and node label, add all node to Graph in advance
    for i in range(2690):
        networkGraph.add_node(i, label = str(i))

    for index, row in G.iterrows():
        row = row.fillna('')
        timestep = ''
        if(row["JIAOYISHIJIAN"] != ''):
            timestep = row["JIAOYISHIJIAN"]
            
        fs0vec = np.zeros(JIAOYIFANGSHI0Array.shape[0])
        fs1vec = np.zeros(JIAOYIFANGSHI1Array.shape[0])
        fs2vec = np.zeros(JIAOYIFANGSHI2Array.shape[0])
        if(row["JIAOYIFANGSHI"] != ''):
            if(len(row["JIAOYIFANGSHI"].split('-')) == 3):
                fs0, fs1, fs2 = row["JIAOYIFANGSHI"].split('-')
            if(len(row["JIAOYIFANGSHI"].split('-')) == 2):
                fs0, fs1= row["JIAOYIFANGSHI"].split('-')
                fs2 = ''
                
            p0 = np.where(JIAOYIFANGSHI0Array==fs0)
            fs0vec[p0] = 1
            p1 = np.where(JIAOYIFANGSHI1Array==fs1)
            fs1vec[p1] = 1
            p2 = np.where(JIAOYIFANGSHI2Array==fs2)
            fs2vec[p2] = 1
            
        sendAccount = "-1"
        if(row["FUKUANZHANGHAO"] != ''):
            sendAccount = row["FUKUANZHANGHAO"]
        
        bankvec = np.zeros(FUKUANFANGYINHANGArray.shape[0])
        if(row["FUKUANFANGYINHANG"] != ""):
            bank = row["FUKUANFANGYINHANG"]
            p = np.where(FUKUANFANGYINHANGArray == bank)
            bankvec[p] = 1
        
        receiveAccount = "-1"
        if(row["SHOUKUANFANGZHANGHAO"] != ""):
            receiveAccount = row["SHOUKUANFANGZHANGHAO"]
        
        moneyTypevec = np.zeros(HUOBIMINGCHENGArray.shape[0])
        if(row["HUOBIMINGCHENG"] != ""):
            moneyType = row["HUOBIMINGCHENG"]
            p = np.where(HUOBIMINGCHENGArray == moneyType)
            moneyTypevec[p] = 1
        
        moneyTotal = "0"
        if(row["YUANBIJINE"] != ""):
            moneyTotal = row["YUANBIJINE"]
        
        cityvec = np.zeros(len(cityList))
        if(row["JIAYIFASHENGDI"] != ""):
            transcity = row['JIAYIFASHENGDI']
            for i,city in enumerate(cityList):
                if(city in transcity):
                    cityvec[i] = 1

        # feature 值统计：
        # len = 1041
        # fs0vec, len=3 ,100占绝对主流
        # fs1vec, len=4, 1000占绝对主流
        # fs2vec, len=13,
        # bankvec,len=676,
        # moneyTypevec,len=2,10占绝对主流
        # cityvec,len=343
        vec = np.concatenate([fs0vec, fs1vec, fs2vec, bankvec, moneyTypevec, cityvec])
        # print(index,timestep, sendAccount, receiveAccount, moneyTotal, vec)
        # networkGraph.add_node(int(sendAccount), label = str(sendAccount))
        # networkGraph.add_node(int(receiveAccount), label = str(receiveAccount))
        networkGraph.add_edge(int(sendAccount), int(receiveAccount), datetime=str(timestep), weight=moneyTotal, feature = str(list(vec)))
        
    # setp 3: write final result
    print(networkGraph)
    nx.write_gml(networkGraph, "moneyLunderingGraphGroundTruth.gml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphFile", default='C:\\Users\\77388\\Desktop\\output.xlsx', type=str)
    args = parser.parse_args()
    
    buildMoneyLaunderingGraph(args.graphFile)

if __name__ == "__main__":
    main()
    
    