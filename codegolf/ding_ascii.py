import math
# num = 12898507791303110870976380826826864066297515407168818640769885558891093839406593468885873956249645794596123834981066692219084096960685665107334282880813426004808892998138324765421764303637530011370253393312790344598368989967413246783243476330757514732431912240841029048040458
# num = 135069356035885703758455496367568992831881386358315477406870580250043130817942674364336076530994511578939389897291271225737615424747050435866983496005770400719100371815194408795514870284998788685348911941501406032695843211597198322101246461577913780411830790457
# num = 16633678475060078900007978045304587303694101970167119251340973070180730471918960922136349523302155314744347721098208113663770116457646709048753721043323569249808834394305725194087039007803869633182252410428142705017272858125859459633403204975601544

num = 5118596130708942111450035950221085501808986470308194971656049325106053607259639375251443701770389178664
num = 546035741070730965825259276435379579366000377703384550597794678437417586986693807945432
num = 3711554526516270318440127420937508751998143598737157433096597706524968
num = 53551514207408255008385712231245455989718943700764968
num = 141109442748753014070536530623875368
num = 6243576359952053976
# num = 1422009388
# num = 34712
# num = 116
num2 = 37710**2-(186**2+(11**2-5))

print(num2)
print(float(num))
# print("test", (num - 47565023539487505127571456**4 ))
# for i in range(2, 100):
	# print(i, math.log(num)/math.log(i))
m_rest = num

for i in range(2, 2000):
	base = round(num**(1/i))
	rest = num - base**i
	if abs(rest) < abs(m_rest):
		m_rest = rest
		print(i, rest, base, num - base**i)
