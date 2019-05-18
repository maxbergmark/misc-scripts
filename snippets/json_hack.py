from matplotlib import pyplot as plt

def rotate(s):
	r = ""
	for i, char in enumerate(s):
		d = -127 - 0*(i%2 == 0) - 0*(i%5 == 4)
		o = ord(char)
		o = (o+d-32) % (127-32) + 32
		r += chr(o)
	return r

b = "ú£Àññ×äóòèîï£»£²¯±¯µ£­£Íàïæôàæä£»£òä£­£ÀòêäåÕîÓàõä£»õóôä­£Æäìò£»³±­£ÆäìòÍèçäõèìä£»°³³±­£ÔòäóÂîïòäïõ£»çàíòä­£Òäõõèïæò£»ú£òçù£»õóôä­£ìôòèâ£»õóôä­£àïèìàõèîïò£»õóôä­£çãÂîïïäâõäå£»çàíòä­£ïîõèçèâàõèîïò£»õóôä­£ûäïÌîåä£»çàíòä­£àôõîÒâóîíí£»çàíòä­£óàõäÆàìä£»çàíòä­£ðôîõäò£»çàíòä­£òçùÞóäöàóå£»õóôä­£ìôòèâÞóäöàóå£»çàíòä­£àïèìàõèîïòÞóäöàóå£»çàíòä­£ûäïÌîåäÞóäöàóå£»çàíòä­£àôõîÒâóîííÞóäöàóå£»çàíòä­£Þóäöàóå£»çàíòä­£ðôîõäòÞóäöàóå£»çàíòä­£õàñÞòèûäÞóäöàóå£»çàíòäü­£ÒäõõèïæòÇíîàõò£»ú£òçù£»±¯±­£ìôòèâ£»±¯±­£ãàâêæóîôïåÌôòèâ£»±¯±­£õàñÞòèûä£»°¯±­£õàñÞåôóàõèîï£»±¯³ü­£ÅàõàÂîïòäïõ£»ú£óäðôèóäå£»õóôä­£àòêäå£»õóôä­£àïàíøõèâò£»õóôä­£àå÷äóõèòèïæ£»õóôä­£ìîñôã£»çàíòäü­£ÂôóóäïõÕéäìä£»µ­£ÂôóóäïõÖîóíå£»±­£ÔïíîâêäåÖîóíåò£»Úõóôä­çàíòäÜ­£ÕôõîóèàíÑóîæóäòò£»¶­£ÖîóíåòÌäïôÔïíîâêäå£»çàíòä­£Ûîîì£»³­£Òà÷ä×äóòèîï£»³±­£Ñóäòõèæä×àíôäò£»ú£±£»°°²¯±ü­£ÑíàøÇàãÈÅ£»£¸Äµ¸±·ÇÀ·±Ã´¸Ä±Â£­£ÇîóâäÑôòé£»ú£±£»çàíòäü­£ÂíäàóÒäõõèïæòÓäöàóåò£»çàíòäü"
true_b = r'{"AppVersion":"3.0.4","Language":"se","AskedToRate":true,"Gems":42,'
#          {$ArrXetsiop$<$30206$.$Napguage$<$se$.$AskefVoTave$<vtue.$Gems$<42.$GemsNihevime$<1442.$UsetCopsepv$<hanse.$Sevvipgs$<{$shz$<vtue.$music$<vtue.$apimaviops$<vtue.$hdCop

# b = "ú£Àññ×äóòèîï£»£±£­£ÀòêäåÕîÓàõä£»çàíòä­£ÃàííÒñàöïäóò£»Ú±­±­±­±­±­±­±­±­±­±Ü­£Âôóóäïâø£»±­£ÂôóóäïâøÅîôãíä£»°³´¶¹¸²´²¹¯¸´±¹³°µ­£ÂôóóäïâøÒäòòèîï£»°·´¸±¸±±³²¯³·°°³±·­£ÂôóóäïâøÍèçäõèìä£»²³·³±´·³°·¯²³¶¶¸³³­£ÂôóóäïâøÓàõä£»±­£ÂôóóäïâøÓàõäÅîôãíä£»³·¶·¶·¯¸µ´±µ·°²³µ¸­£ÂôóóäïâøÓàõäò£»ÚÜ­£ÂôóóäïâøÓàõäòÅîôãíä£»Ú´°¹³³¯±±´·¸¶¶²¹µ­´³³³·´¯´·³µ¶²±¸³·¸­³°±´´·¯²·°·°·¸´¶µ°­³µ±¶±±¯···³´¹²¹¹°°­±¯±­¶±°±¯²±µ²¹¶´·´·±µ¸­¹°¹¸¹¹¯°µ±°±²°³²²¹­·¶¹´¹¯±¶³¸±µ¸³··¹°­µ·µ´°·¯±¹±¹´·¹µ³°°­µ¹°¸¯´¹²¹¸¸³±²²±±´­±¯±­¸¸µ±µ¹¯´¶¹²±·³¹¸¶µ­³¹³¹°°¯´°²¹¹¸°²·²³­²µ¶·¶°¯µ²²¸²¹°¹µµ·­±¯±­±¯±­¹±³¸µ¸¯¸¹·¹±³·²¶°°­³¶¹´°¹¯¶µ³°²µ±¸¹²°­³¶¶µ²²¯¹°¶µ¹·³³°·°­´³°µ±¯°¶¶´¸±³±±µ¹¹­±¯±­¶¶°¸³²¯°²´¸²µ·¹´´¸­µ³¸±¸µ¯³³··´µ³¹³¸´­³´´´·¸¯²¹¸³²³¸µ¶·¶­´³¹¶¶¯¶³³¹¹°¹°°µ¸·­±¯±­·¹²·¸¸¯¶°²°¸°±·°µ­³±²±µ´¯²³²¹¸¶°¸³··­°¸¸¶±·¯±²³³·³µ³±µ·­°±³¹°¯¶¶¹¸¹µ¸·¶±µ°Ü­£Æäìò£»±­£ÆäìòÍèçäõèìä£»±­£Íä÷äíò£»Ú°­±­±­±­±­±­±­±­±­±Ü­£ÕôõîóèàíÂîìñíäõäå£»çàíòä­£ÔòäóÂîïòäïõ£»çàíòä­£ÔñæóàåäÊäøò£»ú£òääå£»±­£¶·¹ãâãç´±ç±ãà°à·£»³´­£¶·¹ãâãç´±ç±ãà°à·Þ°£»³³­£¶³··´µ°°±à¸¹äçµ±£»³±­£¶³··´µ°°±à¸¹äçµ±Þ°£»°¸­£¹à°²¶¹±ã¶µâ´à¸·´£»°±­£¹à°²¶¹±ã¶µâ´à¸·´Þ°£»¸­£µ²µµå¹¶ããâ¹¹°·´ã£»´­£µ²µµå¹¶ããâ¹¹°·´ãÞ°£»³­£ã³¶ç¹·å·²´µ±²µ·¸£»±­£ã³¶ç¹·å·²´µ±²µ·¸Þ°£»±­£àçã·¹°°¶°²±çàå³µ£»±­£àçã·¹°°¶°²±çàå³µÞ°£»±­£å´³µã¶¶ã±µ°²âä·¶£»±­£å´³µã¶¶ã±µ°²âä·¶Þ°£»±­£ä¹ç±äµ°¶¹àåç¸´¹ç£»±­£ä¹ç±äµ°¶¹àåç¸´¹çÞ°£»±­£·çââ´¸±¹âàâ¸±·¸à£»±­£·çââ´¸±¹âàâ¸±·¸àÞ°£»±­£¸å±·ã··à´±äµä²¶²£»±­£¸å±·ã··à´±äµä²¶²Þ°£»±­£´åµ¸³·¶ã´¸¶±¶ãâ¹£»³±­£à±°±µ²²¸àâ¹··å¹à£»°²­£ââ´àâ±µ´ç´´ã²µç¸£»°­£¸µ¹¹ãç±ãäâ±ä±àà´£»°­£´¶ã±âµä±ç¹åãå¸ã±£»±­£°µå·¶²±³²ãåâ±±¶ç£»±­£â¸âä²åàåâ³²´å´ç¸£»±­£¹±°ççâå±åå±ãââ·¶£»±­£¶ç±àåà·¸±²±¹¶à¹¶£»±­£â¶±ã²åà¹ã°¸·à¹¹°£»±­£¶µäâ°³±åàâ±ç±â¸¸£»¸­£¸¸¶´âµ±°¶±³âââçå£»°­£·µ±²·¶°â°°°ã·â´²£»±­£¹â³ãäã¶°µ·´âåâä¶£»±ü­£Òäõõèïæò£»ú£òçù£»õóôä­£ìôòèâ£»õóôä­£çãÂîïïäâõäå£»çàíòä­£ïîõèçèâàõèîïò£»çàíòäü­£ÕèìäÍàòõÒà÷äå£»£³±°¸¬±²¬±¹Õ³±»´·»°´¯¹´±³³¸Û£­£ÕèìäÍàòõÂîííäâõäå£»£³±°¸¬±²¬±¹Õ³±»µ¹»°·¯²¹·³´·Û£­£Óä÷äïôäÕèìäó£»£³±°¸¬±²¬±·Õ°¸»±¸»±³¯±¸°±±µÛ£­£Õèìäóò£»ú£òñääå£»£³±°¸¬±²¬±¶Õ°¸»±¸»±³¯±¸´¸³´Û£­£íàòõÞíîæèï£»£³±°¸¬±²¬±¹Õ³±»µµ»°µ¯¶´±¸¶²Û£­£çóääÆäìò£»£³±°¸¬±²¬±¹Õ°¸»´±»°°¯·¹¹¶µ·Û£­£ñîöäóôñÈïâîìä£»£³±°¸¬±²¬±¹Õ³°»´±»³´¯´²µ°³³Û£­£ñîöäóôñÒñääåÒñàöïäóò£»£³±°¸¬±²¬±¹Õ°²»°²»³¸¯²²°¸´Û£­£ñîöäóôñÆóà÷èõø£»£³±°¸¬±²¬±¹Õ°³»µ°»²¸¯°µ³·´´Û£­£ñîöäóôñÂéàïâäÅäòõóîø£»£³±°¸¬±²¬±¹Õ°³»²¸»µ±¯³¹¹¸¹µÛ£ü­£ÃàííòÅàõà£»ÚÜ­£ÃàííòÅàõàÒèìñíä£»ÚÜ­£ÂôóóäïõÕéäìä£»±­£Ãíîâêàæäò£»ÚÜ­£ÕôõîóèàíÑóîæóäòò£»±­£Ñóäòõèæä×àíôä£»°°²¯±­£Òâóîíí×àíôä£»¬°²¹¯³µ±¶²¹ü"
# true_b = r'{"AppVersion":"0","'
n = len(true_b)
print(ord("»") - ord(":"))
print(ord("£") - ord("\""))
print(ord("ú") - ord("{"))
# r_a = rotate(a)
# print(r_a)
r_b = rotate(b)
print(r_b)

diffs = [ord(r_b[i]) - ord(true_b[i]) for i in range(n)]
for d in diffs:
	print(d)

plt.plot(diffs)
plt.show()