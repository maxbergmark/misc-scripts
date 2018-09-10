import numpy as np
from matplotlib import pyplot as plt

inp = np.array([0.36407017707825, 0.47909688949585, 0.29213190078735, 0.37814998626709, 0.0092868804931641, 0.010026931762695, 0.0095350742340088, 0.010051965713501, 0.0088129043579102, 0.0088140964508057, 0.0087800025939941, 0.0089409351348877, 0.008868932723999, 0.16152000427246, 0.15154504776001, 0.0094399452209473, 0.15007090568542, 0.0085258483886719, 0.14718508720398, 0.0085990428924561, 0.0087690353393555, 0.40525698661804, 0.43818211555481, 0.0094859600067139, 0.008969783782959, 0.0087289810180664, 0.0085279941558838, 0.0085301399230957, 0.0084497928619385, 0.0085699558258057, 0.0097482204437256, 0.0086600780487061, 0.0086739063262939, 0.3621940612793, 0.42150402069092, 0.0086691379547119, 0.0092728137969971, 0.0085709095001221, 0.008465051651001, 0.045187950134277, 0.045979976654053, 0.32756090164185, 0.052535057067871, 0.35166907310486, 0.0088319778442383, 0.0085599422454834, 0.0087571144104004, 0.37023186683655, 0.0085048675537109, 0.008713960647583, 0.0087859630584717, 0.18121194839478, 0.47454190254211, 0.48651885986328, 0.49077582359314, 0.47754001617432, 0.48115706443787, 0.48788499832153, 0.48117804527283, 0.48144102096558, 0.48267793655396, 0.48497414588928, 0.74352383613586, 0.010401010513306, 0.050014019012451, 0.0097579956054688, 0.0086197853088379, 0.22370100021362, 0.42306995391846, 0.41884803771973, 0.42586016654968, 0.41860699653625, 0.41505002975464, 0.42643404006958, 0.41968703269958, 0.035868883132935, 0.0089530944824219, 0.0093660354614258, 0.0086309909820557, 0.24960398674011, 0.0087277889251709, 0.008674144744873, 0.0089619159698486, 0.052428007125854, 0.0086839199066162, 0.0083739757537842, 0.0092017650604248, 0.0086119174957275, 0.0086169242858887, 0.42403221130371, 0.42466115951538, 0.0098750591278076, 0.010623931884766, 0.0084648132324219, 0.0088639259338379, 0.32536792755127, 0.00856614112854, 0.27268695831299, 0.25495886802673, 0.26520299911499, 0.48134899139404, 0.26705503463745, 0.48448300361633, 0.0088720321655273, 0.21780705451965, 0.37839198112488, 0.21526789665222, 0.43622303009033, 0.22095489501953, 0.47828006744385, 0.48358201980591, 0.48763608932495, 0.48902988433838, 0.48134112358093, 0.48169612884521, 0.48564386367798, 0.47773003578186, 0.47321319580078, 0.48082399368286, 0.47640895843506, 0.4876880645752, 0.21119594573975, 0.48250484466553, 0.0092799663543701, 0.0088660717010498, 0.21215677261353, 0.0086448192596436, 0.0087399482727051, 0.009321928024292, 0.21023201942444, 0.48190498352051, 0.0087800025939941, 0.19996881484985, 0.49165296554565, 0.48110103607178, 0.48062109947205, 0.48616003990173, 0.48318290710449, 0.48282885551453, 0.0089728832244873, 0.21323895454407, 0.0087881088256836, 0.010098218917847, 0.0087099075317383, 0.0089969635009766, 0.0088119506835938, 0.19992709159851, 0.48667001724243, 0.20080208778381, 0.19970011711121, 0.19128704071045, 0.0091760158538818, 0.0090479850769043, 0.029571056365967, 0.028800010681152, 0.0090310573577881, 0.14502501487732, 0.19484114646912, 0.023140907287598, 0.020947217941284, 0.021481990814209, 0.021965026855469, 0.021106004714966, 0.021531820297241, 0.021970987319946, 0.021611928939819, 0.0225989818573, 0.1510808467865, 0.0086290836334229, 0.0086870193481445, 0.0089700222015381, 0.0086429119110107, 0.037080049514771, 0.0088350772857666, 0.19647812843323, 0.48103499412537, 0.48642086982727, 0.48221397399902, 0.47692203521729, 0.48303008079529, 0.47784304618835, 0.61584115028381, 0.47550797462463, 0.48867106437683, 0.47698497772217, 0.020917177200317, 0.0088889598846436, 0.008397102355957, 0.0083680152893066, 0.1447069644928, 0.020258188247681, 0.14676117897034, 0.41070508956909, 0.14818906784058, 0.14716410636902, 0.1411759853363, 0.019475221633911, 0.0083639621734619, 0.022527933120728, 0.16906714439392, 0.11619877815247, 0.18751287460327, 0.0094001293182373, 0.0088989734649658, 0.0084810256958008, 0.18384289741516, 0.48027300834656, 0.18437600135803, 0.17716121673584, 0.0086710453033447, 0.0091249942779541, 0.0086119174957275, 0.0086791515350342, 0.17370200157166, 0.0085628032684326, 0.17875409126282, 0.0095999240875244, 0.4838171005249, 0.0087690353393555, 0.16691589355469, 0.0090069770812988, 0.019314050674438, 0.056759119033813, 0.0091700553894043, 0.0092778205871582, 0.0091109275817871, 0.0090751647949219, 0.0085320472717285, 0.008781909942627, 0.0090630054473877, 0.0087251663208008, 0.0088138580322266, 0.0089049339294434, 0.0088639259338379, 0.0087449550628662, 0.0091941356658936, 0.16646194458008, 0.13347697257996, 0.13720512390137, 0.0090088844299316, 0.010601997375488, 0.0089371204376221, 0.0085780620574951, 0.0085718631744385, 0.012150049209595, 0.0086591243743896, 0.0091440677642822, 0.00862717628479, 0.16121292114258, 0.0087659358978271, 0.1606388092041, 0.008598804473877, 0.0089559555053711, 0.0086088180541992, 0.0086569786071777, 0.15517687797546, 0.00838303565979, 0.0087070465087891, 0.0085439682006836, 0.0089480876922607, 0.0087051391601562, 0.0090880393981934, 0.26013612747192, 0.26186299324036, 0.008991003036499, 0.0088050365447998, 0.2400598526001, 0.0083620548248291, 0.008275032043457, 0.0081648826599121, 0.42616009712219, 0.0080251693725586, 0.43554186820984, 0.0095040798187256, 0.0090658664703369, 0.0086169242858887, 0.008807897567749, 0.26185703277588, 0.0086500644683838, 0.010586977005005, 0.0090229511260986, 0.0086920261383057, 0.029325008392334, 0.027107000350952, 0.0087990760803223, 0.008774995803833, 0.0088770389556885, 0.0087840557098389, 0.0098180770874023, 0.0087010860443115, 0.37315392494202, 0.35053992271423, 0.0092959403991699, 0.0090560913085938, 0.01009202003479, 0.33657503128052, 0.33322691917419, 0.52888894081116, 0.47778105735779, 0.37369704246521, 0.48234605789185, 0.47687792778015, 0.011431932449341, 0.0086500644683838, 0.0085620880126953, 0.0098018646240234, 0.0084059238433838, 0.0088708400726318, 0.0086369514465332, 0.40053415298462, 0.48920798301697, 0.73865103721619, 0.0084719657897949, 0.0091660022735596, 0.010235071182251, 0.31746292114258, 0.42716002464294, 0.38389492034912, 0.0090601444244385, 0.0088920593261719, 0.008742094039917, 0.37229418754578, 0.48100185394287, 0.33596396446228, 0.48903799057007, 0.30558204650879, 0.43349504470825, 0.43475890159607, 0.0086040496826172, 0.008836030960083, 0.47895979881287, 0.47913217544556, 0.31633901596069, 0.48091387748718, 0.28233885765076, 0.42120385169983, 0.3298761844635, 0.49269986152649, 0.009005069732666, 0.0094420909881592, 0.0090010166168213, 0.0092999935150146, 0.0087919235229492, 0.0086739063262939, 0.0096399784088135, 0.0086090564727783, 0.0085608959197998, 0.0084071159362793, 0.0093619823455811, 0.013783931732178, 0.0084629058837891, 0.0084359645843506, 0.0085899829864502, 0.0088109970092773, 0.0083920955657959, 0.0084569454193115, 0.0087978839874268, 0.0090610980987549, 0.0089380741119385, 0.0086629390716553, 0.0091178417205811, 0.0088798999786377, 0.00885009765625, 0.0088021755218506, 0.0087680816650391, 0.0087618827819824, 0.0085518360137939, 0.0086209774017334, 0.0083611011505127, 0.0085530281066895, 0.0084939002990723, 0.0083801746368408, 0.0087170600891113, 0.0088770389556885, 0.0084619522094727, 0.008868932723999, 0.0089099407196045, 0.0089809894561768, 0.0089600086212158, 0.008573055267334, 0.0089030265808105, 0.0085709095001221, 0.0084989070892334, 0.0087130069732666, 0.0087311267852783, 0.0087840557098389, 0.0097019672393799, 0.0086691379547119, 0.0086128711700439, 0.0090200901031494, 0.0089068412780762, 0.0086910724639893, 0.010078907012939, 0.0088479518890381, 0.0085780620574951, 0.0091769695281982, 0.0091328620910645, 0.0087368488311768, 0.0091540813446045, 0.0087921619415283, 0.0088889598846436, 0.0087189674377441, 0.0086331367492676, 0.0085129737854004, 0.0087499618530273, 0.0086870193481445, 0.00895094871521, 0.0089480876922607, 0.0083351135253906, 0.0085470676422119, 0.0086719989776611, 0.142737865448, 0.0085568428039551, 0.17576599121094, 0.0082948207855225, 0.0086019039154053, 0.0084400177001953, 0.008803129196167, 0.0086228847503662, 0.0087771415710449, 0.0087661743164062, 0.0086970329284668, 0.0084531307220459, 0.0087809562683105, 0.14536118507385, 0.1522490978241, 0.0091719627380371, 0.14203310012817, 0.14971208572388, 0.0085849761962891, 0.0087220668792725, 0.0089030265808105, 0.1388988494873, 0.14458799362183, 0.13753604888916, 0.0087020397186279, 0.0087409019470215, 0.0083320140838623, 0.0086069107055664, 0.0086071491241455, 0.14098310470581, 0.0090141296386719, 0.15807199478149, 0.15155410766602, 0.14060306549072, 0.0084471702575684, 0.0089569091796875, 0.0087838172912598, 0.0087349414825439, 0.0087440013885498, 0.13383984565735, 0.13402485847473, 0.008897066116333, 0.12573313713074, 0.12768602371216, 0.12520384788513, 0.008573055267334, 0.13057279586792, 0.0086688995361328, 0.12677407264709, 0.12229418754578, 0.12068295478821, 0.0090830326080322, 0.0087080001831055, 0.34467506408691, 0.0086531639099121, 0.32276701927185, 0.0087130069732666, 0.0084869861602783, 0.33576798439026, 0.0089180469512939, 0.34055876731873, 0.0086381435394287, 0.010305881500244, 0.0087862014770508, 0.0089731216430664, 0.0086510181427002, 0.0087301731109619, 0.0085179805755615, 0.0086901187896729, 0.0085580348968506, 0.0088508129119873, 0.0086889266967773, 0.0086140632629395, 0.0086789131164551, 0.0092260837554932, 0.008587121963501, 0.0084929466247559, 0.0088100433349609, 0.0087368488311768, 0.009235143661499, 0.0086331367492676, 0.32408404350281, 0.42553186416626, 0.4333438873291, 0.0087900161743164, 0.0084071159362793, 0.0088257789611816, 0.0089108943939209, 0.0087430477142334, 0.0085139274597168, 0.0084919929504395, 0.0089471340179443, 0.0088720321655273, 0.0088880062103271, 0.30812191963196, 0.30635786056519, 0.0086779594421387, 0.0086379051208496, 0.008652925491333, 0.0089840888977051, 0.0086860656738281, 0.0092649459838867, 0.0086390972137451, 0.0096230506896973, 0.0087800025939941, 0.0095829963684082, 0.0088291168212891, 0.0089831352233887, 0.0083990097045898, 0.34174084663391, 0.0087440013885498, 0.34072399139404, 0.31689596176147, 0.0086197853088379, 0.33790111541748, 0.33595299720764, 0.48137283325195, 0.0087821483612061, 0.0085690021514893, 0.32003211975098, 0.0092799663543701, 0.008756160736084, 0.42200303077698, 0.0088880062103271, 0.0087780952453613, 0.48516297340393, 0.45382809638977, 0.42753887176514, 0.4543240070343, 0.009066104888916, 0.2721529006958, 0.44496297836304, 0.4217529296875, 0.30284094810486, 0.0086159706115723, 0.24166989326477, 0.0086369514465332, 0.0086910724639893, 0.0087790489196777, 0.008864164352417, 0.0092029571533203, 0.23467803001404, 0.008530855178833, 0.0089590549468994, 0.0092759132385254, 0.0088260173797607, 0.0090038776397705, 0.0086400508880615, 0.2257981300354, 0.27276921272278, 0.48206281661987, 0.40386390686035, 0.72667193412781, 0.0084829330444336, 0.20590496063232, 0.0089459419250488, 0.034172058105469, 0.067356824874878, 0.068289041519165, 0.069056987762451, 0.19856595993042, 0.47428679466248, 0.30024290084839, 0.58061718940735, 0.37009286880493, 0.37084102630615, 0.37398195266724, 0.36677193641663, 0.36775517463684, 0.2004930973053, 0.38266706466675, 0.20773506164551, 0.40141820907593, 0.31758999824524, 0.0089600086212158, 0.0089490413665771, 0.069785833358765, 0.79718017578125, 0.74958491325378, 0.75209403038025, 0.74131298065186, 0.73835587501526, 0.74431109428406, 0.0090539455413818, 0.33641600608826, 0.35454201698303, 0.0086281299591064, 0.0085399150848389, 0.0089108943939209, 0.0087289810180664, 0.0090160369873047, 0.0093569755554199, 0.0087909698486328, 0.0084660053253174, 0.0089190006256104, 0.0089149475097656, 0.0087399482727051, 0.008965015411377, 0.0088391304016113, 0.0084657669067383, 0.0088980197906494, 0.0088741779327393, 0.0085380077362061, 0.0099270343780518, 0.0089640617370605, 0.0083329677581787, 0.0088250637054443, 0.008552074432373, 0.0088489055633545, 0.0096569061279297, 0.0086708068847656, 0.008598804473877, 0.0086960792541504, 0.26372790336609, 0.47407793998718, 0.27913904190063, 0.46211504936218, 0.0087671279907227, 0.0087289810180664, 0.15474700927734, 0.15417695045471, 0.4253830909729, 0.42493987083435, 0.1511218547821, 0.3793671131134, 0.53324103355408, 0.0088231563568115, 0.0089471340179443, 0.0089361667633057, 0.008965015411377, 0.0089631080627441, 0.1966290473938, 0.19351100921631, 0.008620023727417, 0.28383207321167, 0.030603170394897, 0.029333829879761, 0.0088639259338379, 0.0086579322814941, 0.0092101097106934, 0.44718813896179, 0.44798684120178, 0.41761708259583, 0.42681384086609, 0.0088820457458496, 0.0089859962463379, 0.0091469287872314, 0.44048285484314, 0.41821002960205, 0.20066809654236, 0.0091869831085205, 0.20324802398682, 0.18763303756714, 0.0089151859283447, 0.031677007675171, 0.44667601585388, 0.44472503662109, 0.45451498031616, 0.4584538936615, 0.45526599884033, 0.44487190246582, 0.44349479675293, 0.44077110290527, 0.4507908821106, 0.4547758102417, 0.44795298576355, 0.45242309570312, 0.45271801948547, 0.44903779029846, 0.44553208351135, 0.45232105255127, 0.45088696479797, 0.36505103111267, 0.37113404273987, 0.48470687866211, 0.48045086860657, 0.47835206985474, 0.0086469650268555, 0.44758105278015, 0.45362997055054, 0.0094530582427979, 0.060402870178223, 0.36271119117737, 0.0091269016265869, 0.0087008476257324, 0.36854004859924, 0.48014402389526, 0.0093541145324707, 0.42294001579285, 0.21760201454163, 0.43084192276001, 0.032296895980835, 0.22108912467957, 0.0092129707336426, 0.0091278553009033, 0.43526411056519, 0.008922815322876, 0.0091469287872314, 0.0091118812561035, 0.0094139575958252, 0.0098497867584229, 0.02954888343811, 0.43049693107605, 0.44618892669678, 0.4470100402832, 0.43715405464172, 0.032850980758667, 0.40336298942566, 0.91808700561523, 0.95286989212036, 0.97219300270081, 0.94522404670715, 0.4793860912323, 0.47919011116028, 0.43560099601746, 0.44183397293091, 0.4433000087738, 0.44619989395142, 0.45832896232605, 0.44482707977295, 0.45387411117554, 0.4433798789978, 0.0089890956878662, 0.0088260173797607, 0.0088570117950439, 0.0094060897827148, 0.35164999961853, 0.43076801300049, 0.46555519104004, 0.37809705734253, 0.47432708740234, 0.4671938419342, 0.39430499076843, 0.43767905235291, 0.43819403648376, 0.35464715957642, 0.47019791603088, 0.47870206832886, 0.46796917915344, 0.46728301048279, 0.63931608200073, 0.41428303718567, 0.46527314186096, 0.39416694641113, 0.47054100036621, 0.040148019790649, 0.064403057098389, 0.06219220161438, 0.063762903213501, 0.40462398529053, 0.30773401260376, 0.46629118919373, 0.15176892280579, 0.23872089385986, 0.23801493644714, 0.0090081691741943, 0.0086748600006104, 0.30406093597412, 0.45968508720398, 0.14179682731628, 0.26135587692261, 0.13737511634827, 0.26125001907349, 0.44897413253784, 0.50648212432861, 0.0096769332885742, 0.008969783782959, 0.1966609954834, 0.24684000015259, 0.0087540149688721, 0.2964129447937, 0.59037089347839, 0.46952390670776, 0.18258714675903, 0.53712105751038, 0.11929321289062, 0.0088300704956055, 0.4142529964447, 0.011073112487793, 0.0088810920715332, 0.0091118812561035, 0.0088441371917725, 0.0092298984527588, 0.0087878704071045, 0.0087430477142334, 0.0089030265808105, 0.0092828273773193, 0.015635967254639, 0.0092802047729492, 0.0097248554229736, 0.046235084533691, 0.0091419219970703, 0.046344995498657, 0.045109987258911, 0.0088229179382324, 0.0093581676483154, 0.0089590549468994, 0.0085000991821289, 0.0085899829864502, 0.0089859962463379, 0.0087900161743164, 0.0087637901306152, 0.045545101165771, 0.42518591880798, 0.68967604637146, 0.0093698501586914, 0.0087862014770508, 0.0089430809020996, 0.055779933929443, 0.04118800163269, 0.041238069534302, 0.0090060234069824, 0.0089719295501709, 0.0091030597686768, 0.0088629722595215, 0.041743993759155, 0.068326950073242, 0.067699909210205, 0.036611080169678, 0.036588191986084, 0.35367894172668, 0.036565065383911, 0.0093240737915039, 0.0085101127624512, 0.0094950199127197, 0.0086531639099121, 0.0097367763519287, 0.040817975997925, 0.00895094871521, 0.0087370872497559, 0.039878845214844, 0.0086328983306885, 0.0089068412780762, 0.042190074920654, 0.0087990760803223, 0.0089190006256104, 0.25142312049866, 0.2488579750061, 0.4570050239563, 0.4434928894043, 0.44692802429199, 0.44448804855347, 0.44751906394958, 0.0088739395141602, 0.44109201431274, 0.0089538097381592, 0.0088059902191162, 0.21069407463074, 0.47201991081238, 0.034089088439941, 0.066972970962524, 0.032440185546875, 0.078376054763794, 0.0090320110321045, 0.0090038776397705, 0.057153940200806, 0.0091350078582764, 0.083742141723633, 0.097939968109131, 0.48180818557739, 0.064941167831421, 0.1290819644928, 0.064385890960693, 0.36797499656677, 0.48208999633789, 0.6481990814209, 0.60416412353516, 0.87715601921082, 0.42468619346619, 0.058233976364136, 0.064166069030762, 0.05476188659668, 0.35758805274963, 0.31650996208191, 0.34782910346985, 0.74572801589966, 0.0095589160919189, 0.33976697921753, 0.55681705474854, 0.77826690673828, 0.57665395736694, 0.4835000038147, 0.36300802230835, 0.35570788383484, 0.49130320549011, 0.3393931388855, 0.43255710601807, 0.31989789009094, 0.3121919631958, 0.44308304786682, 0.31449413299561, 0.246248960495, 0.0095810890197754, 0.41510391235352, 0.25603199005127, 0.26425814628601, 0.57241487503052, 0.71192312240601, 0.70226812362671, 0.74986100196838, 0.75585293769836, 0.57004380226135, 0.47921800613403, 0.0093801021575928, 0.50134301185608, 0.19113183021545, 0.46784090995789, 0.1921751499176, 0.47425007820129, 0.18485593795776, 0.87314200401306, 0.607666015625, 0.47869610786438, 0.28595495223999, 0.73698377609253, 0.18168091773987, 0.60392808914185, 0.45125985145569, 0.47594499588013, 0.2340099811554, 0.16005420684814, 0.47405791282654, 0.46927285194397, 0.15941286087036, 0.46608400344849, 0.43529415130615, 0.41842603683472, 0.21977496147156, 0.74300003051758, 0.7454879283905, 0.76124405860901, 0.15542507171631, 0.47412776947021, 0.77412104606628, 0.67847299575806, 0.47038388252258, 0.52392601966858, 0.47294902801514, 0.85085606575012, 0.47508192062378, 0.4691469669342, 0.65707087516785, 0.47489714622498, 0.47461986541748, 0.47356200218201, 0.57401704788208, 0.59769988059998, 0.46985697746277, 1.2089450359344, 0.64268589019775, 0.64765286445618, 1.3676280975342, 0.50098919868469, 0.68026685714722, 1.0499980449677, 0.6276068687439, 0.19261503219604, 0.47867298126221, 0.15428996086121, 0.44148683547974, 0.14888095855713, 0.62761902809143, 0.76910090446472])

print(inp[inp>0.01].mean())
plt.hist(inp[inp>0.01], 100)
plt.show()

