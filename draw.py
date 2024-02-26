import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import random

local_epochs = 30
aggregate_epoch = 100

teacher_history=[0.757916722494013,0.483362663381239,0.338127978773678,0.258258443404646,0.233608391944099,
                 0.229937193674199,0.231401825049344,0.227518180012702,0.218614104579476,0.22195506299243,
                 0.220898351809557,0.213257326483726,0.219167319992009,0.216314271197599,0.213625468744951,0.21493697194492,0.210152978392208,0.208491923731916,0.206220889631439,0.205412449219647,
                 0.20134172214245,0.20230889531230,0.209943256733572,0.2089162146245672,0.2070323635262458,0.2069345672567890,0.206092316754783,0.205732121145464,0.205412449219647,0.2003452535415632]

student_history=[0.96839622256335,0.687171402173883,0.530207678626565,0.459257708142785,0.43616648606693,
                 0.432830622154123,0.435808385161792,0.433278192506117,0.421864626688115,0.431152415331672,
                 0.429158881355734,0.430165660044726,0.430712494471493,0.423878585885552,0.420534840920392,0.42719214961108,0.429874816417694,0.424320150768055,0.423692352322971,0.423639840112012,
                 0.430639840112012,0.436311452374134,0.434712345241421,0.432324673456789,0.431934523156234,0.431357834567257,0.428457245823481,0.424114578858035,0.423123523156723,0.42031213526795]

student_kd_history=[0.901510426156661,0.5790356508703792,0.4400592018295737,0.34039346359757814,0.2546823030219359,
                    0.232475165829939,0.2309244339606341,0.22731592590668624,0.22880284139128292,0.22239115010990815,
                    0.22515186690302456,0.218275455152287,0.221817807162509, 0.219794731266358, 0.217967504781835, 0.212491775456596, 0.21212405942468, 0.209642034123925, 0.207819674127242, 0.206220889631439,
                    0.206000876621920, 0.215992346135695,0.210002331343672, 0.2094261583468211, 0.2071942456789224, 0.2071235672168231, 0.2061145234562515, 0.205993213424567, 0.205910102119342, 0.200935213452146]


teacher_acc=[0.9004251386321626,0.9215033887861984,0.9299285274183611,0.9375791743684535,0.9403592113370302,
             0.94350462107208873,0.943973567467652495,0.943793518176216882,0.94391953173136167,0.94400574245224893,
             0.944035674676525,0.9449139248305607,0.9441928527418361,0.944550462107208873,0.94435742452248922,0.94474984596426371,0.94439346888478126,0.944513567467652495,0.94483518176216882,0.94462711028958718,
             0.945015033887861984,0.9452816389402341,0.94513518176216882,0.9452760320394331,0.945518176216882,0.9457921749845965,0.9459760320394331,0.9454437461491066,0.9456002464571781,0.9456623536660505]

student_acc=[0.8802711028958718,0.8959630314232902,0.9044984596426371,0.91273321010474,0.9208089956869994,
             0.924953173136167,0.9268798521256932,0.926033887861984,0.92744177449168208,0.9279556377079482,
             0.9273258163894024,0.9280637091805298,0.92789556377079482,0.9282556377079482,0.928328280961183,0.9280009242144177,0.9285500308071473,0.9287444239063462,0.9286652495378928,0.9288276032039433,
             0.9289549599507086,0.9290009242144177,0.92935009242144177,0.9290587800369686,0.92992433148490449,0.92958256315465188, 0.9301163894023414, 0.9309567467652495, 0.9303946395563771, 0.931424460874923]

student_kd_acc=[0.8917843499691928,0.9103419593345656,0.9196500308071473,0.9323419593345656,0.93794959950708565,
                0.94170717806531116,0.9430798521256932,0.9428637091805298,0.9421717806531116,0.942910967344439,
                0.9439735674676525,0.9433040665434381,0.9434258163894024,0.9433032101047443,0.94300735674676525,0.9437009242144177,0.9438609242144177,0.9439049599507086,0.94393089956869994,0.9439089956869994,
                0.94412089956869994,0.9442630314232902,0.944517067159581,0.94477067159581,0.945317067159581,0.945733887861984,0.9456459950708565,0.945123887861984,0.94519933887861984,0.9453132572341362]


ALSGD_acc = [0.8455082, 0.857907, 0.86531394, 0.87448935, 0.8800927, 0.88701917, 0.8944055, 0.8975647, 0.9009619, 0.9052157,
             0.91115993, 0.91639173, 0.9207073, 0.9249668, 0.9272917, 0.930099066, 0.93300855, 0.93506187, 0.9370788967, 0.93806464,
             0.93970085, 0.940408587, 0.94065803, 0.9404625, 0.9403968, 0.94069, 0.94046577,0.9405936, 0.9406996, 0.9406649,
             0.9407878, 0.9407215, 0.94080587, 0.9404383, 0.94062853, 0.94091875, 0.94072136,0.94037933, 0.94084985, 0.9402426,
             0.9406853, 0.94077145, 0.9402099, 0.9403571, 0.9409996, 0.9401508, 0.940384,0.93956457, 0.94065916, 0.94072747,
             0.94054876, 0.94076256, 0.94005766, 0.94077375, 0.9401782, 0.94029421, 0.940363144,0.940474634, 0.94060222, 0.94077373,
             0.940774976, 0.94085776, 0.9409216, 0.9408473, 0.9404132, 0.9409234, 0.94092295,0.9414151, 0.9417764, 0.9411243,
             0.9414107, 0.9414794, 0.94112244, 0.9416263, 0.9412062, 0.9417539, 0.9415333, 0.9418682, 0.9410745, 0.94178805,
             0.9414897, 0.94107176, 0.9412662, 0.9414129, 0.9420407, 0.9422666, 0.94237466, 0.94276894, 0.9420552, 0.94288884,
             0.94242765, 0.9427406, 0.9426509, 0.9429101, 0.94207383, 0.94289156, 0.9428866,0.942158,0.9424356,0.94228556]

FedProx_acc = [0.85012334, 0.8549462, 0.8570308, 0.8607305, 0.864351705, 0.8665076, 0.8705422, 0.87255736, 0.87465356, 0.8775282,
              0.8795142, 0.8832264, 0.88612615, 0.8902326, 0.89438844, 0.89802895, 0.9020706, 0.90604174, 0.9097676, 0.91270376,
              0.914903, 0.9180573, 0.9210953, 0.92408182, 0.926028854, 0.928030496, 0.92988331, 0.931093936, 0.93204089754, 0.933461864,
              0.934063, 0.934852, 0.934562, 0.9345632, 0.9346781, 0.9346892, 0.9343783, 0.9347825, 0.9346568, 0.9345325,
              0.93405625, 0.93408733, 0.93413763, 0.934162, 0.93414671, 0.934146509, 0.93451724, 0.93453348, 0.93456327, 0.93461886,
              0.93462956, 0.934704995, 0.93472667, 0.93405876, 0.93476055, 0.9348746, 0.93493775, 0.9341, 0.9345202, 0.9344641,
              0.9345978, 0.9346293, 0.9344915, 0.9342239, 0.9343218, 0.93475225, 0.9339214, 0.93409913, 0.93406145, 0.93404196,
              0.93425482, 0.934160564, 0.9342667, 0.934167024, 0.9340078, 0.93433709, 0.93425472, 0.934291, 0.9345599, 0.9346443,
              0.93435054, 0.9347949, 0.9348305, 0.9343339, 0.9347993, 0.9349755, 0.9352414, 0.9355823, 0.93556593, 0.935557,
              0.9351096, 0.93579034, 0.9358882, 0.93565255, 0.93526724, 0.9356996, 0.9353128, 0.9357889, 0.93549294, 0.9358525]

FedAvg_acc = [0.84356576, 0.844796814, 0.84608215, 0.8486135, 0.8506245, 0.852824, 0.85455775, 0.8560515, 0.85893586, 0.86044844,
               0.8624852, 0.8646525, 0.86614316, 0.8683086, 0.87025335, 0.87237, 0.87462141, 0.87617996, 0.8796633, 0.8817072,
               0.8839176, 0.88520967, 0.8872135, 0.88951433, 0.891161, 0.8930675, 0.89511904, 0.8977796, 0.8991137, 0.90139627,
               0.90308145, 0.905025148, 0.907021529, 0.9092407216, 0.911036546, 0.9125391, 0.9130856, 0.913660569, 0.91444476, 0.91508557,
               0.916051076, 0.91608487, 0.916430544, 0.9167477, 0.9169786, 0.9168413735, 0.916919504, 0.916527644, 0.916663436, 0.916773634,
               0.91684269, 0.91665854, 0.91659662, 0.916860567, 0.9161956, 0.91657333, 0.91649744, 0.9167327, 0.91697146, 0.91696734,
               0.9160884, 0.91659093883, 0.916092107, 0.91649612, 0.91669907925, 0.91659014935, 0.9166970907, 0.91699121, 0.9163018304, 0.916509049,
               0.916010551, 0.91631569, 0.9165165, 0.91634119, 0.9164672, 0.91677197, 0.91698356, 0.91638613, 0.91679405, 0.91603455,
               0.9166887, 0.9168799, 0.9167479, 0.91744265, 0.91753724, 0.9177084, 0.9177905, 0.91787955, 0.9178467, 0.9174742,
               0.9177073, 0.9172094, 0.91857417, 0.9181403, 0.91842416, 0.91817937, 0.91839365, 0.9184179, 0.91864034, 0.91834875]




ALSGD_loss = [0.9771139, 0.80704224,0.6917365,0.6098699, 0.5529115,0.50211835, 0.4544244, 0.4279254, 0.3967385, 0.35055515,
              0.3206134, 0.29015536, 0.27017934, 0.25819803, 0.24950018, 0.2402217, 0.2308732,0.22307517, 0.2144945, 0.2097536,

              0.19899332, 0.1962787, 0.19473068, 0.19632707, 0.19751904, 0.19641695, 0.1962981, 0.1947934, 0.19226135, 0.1977245,
              0.1965281, 0.19562243, 0.19543685, 0.19433667, 0.1931085, 0.19157552, 0.1908635, 0.1963324, 0.1916587, 0.19159404,
              0.1946541, 0.1994806, 0.19603708, 0.19504103, 0.19954196, 0.19943765, 0.19908116, 0.1965732, 0.19440222, 0.19375768,
              0.19877614, 0.19631247, 0.1962091, 0.19602458, 0.19556017, 0.19441716, 0.1942331, 0.1937303, 0.19521176, 0.1973077,
              0.19977395, 0.193764912, 0.1979643, 0.1938547, 0.1915704, 0.1904987, 0.199373, 0.1976775, 0.19642921, 0.19053598,
              0.19641967, 0.19335024, 0.1922, 0.19088224, 0.1982939, 0.1997798, 0.19047508, 0.1976546, 0.19504902, 0.1914977,
              0.19121555, 0.19639283, 0.19575447, 0.19441706, 0.1928685, 0.194902, 0.1935686, 0.1996803, 0.19040072, 0.198471498,
              0.19589758, 0.1972918, 0.19899161, 0.19744365, 0.19718105, 0.19430676, 0.19451132, 0.19079613, 0.19688516, 0.1997263]

FedAvg_loss = [1.031351,0.9063474, 0.8568089, 0.79209226, 0.7543053, 0.7275452, 0.7076524, 0.6858916, 0.66077377, 0.63131656,
               0.6006413,  0.5748849, 0.554001, 0.5305933, 0.511384, 0.5001536, 0.49078167,0.4765676, 0.46205448, 0.4554464,
               0.44768977, 0.4408313, 0.43650317, 0.4294696, 0.4249926, 0.41945056, 0.41564536, 0.41190815, 0.4077835, 0.40205247,
               0.3988001, 0.39470627, 0.39065274, 0.38609032, 0.38322062, 0.38085515, 0.37707008, 0.37395874, 0.36825136, 0.362673173,

               0.35986308, 0.3587617, 0.3597533, 0.35892256, 0.35753544, 0.35793358, 0.3570152, 0.3513812, 0.35284905, 0.35679968,
               0.35997295, 0.35765743, 0.35650855, 0.35642213, 0.35919816, 0.35723273, 0.35405935, 0.35317783, 0.35634955, 0.3534584,
               0.3531802, 0.35853473, 0.35740556, 0.35603515, 0.35954314, 0.35779172, 0.3568201, 0.35629634, 0.358452677, 0.359441438,
               0.3541339, 0.3516954, 0.35150115, 0.35052327, 0.3536963, 0.35292326, 0.35258933, 0.353116493, 0.35432124, 0.3541575,
               0.35954482, 0.356878, 0.35317884, 0.35913332, 0.35424482, 0.35073592, 0.35920784, 0.35892606, 0.35818517, 0.35653788,
               0.35505932, 0.3559035, 0.35025546, 0.3598996, 0.35923818, 0.35887245, 0.35772306, 0.35625593, 0.35603172, 0.35581974]

FedProx_loss = [0.89880824, 0.7798019, 0.70366535, 0.65715636, 0.6031606, 0.5668496, 0.53844333, 0.5137152, 0.49586807,0.47937494,
                0.4601819, 0.4462123, 0.42047036, 0.408361, 0.39608502, 0.38583045, 0.3756188, 0.36823932, 0.36015403,0.3510611,
                0.3421805, 0.3355158, 0.32831005, 0.31756266, 0.31082464, 0.30214105, 0.29739102, 0.29337738, 0.28895444, 0.283993395,

                0.27977328, 0.27870442, 0.27244146, 0.27885523, 0.2707656, 0.27885512, 0.2768665, 0.27545384,0.27511573, 0.279222,
                0.27555062, 0.27079703, 0.27603977, 0.2772474, 0.27133835, 0.27093748, 0.27558707, 0.27490233,0.2735048, 0.27090435,
                0.2708447, 0.2756367, 0.27926028, 0.27837855, 0.27545676, 0.27346522, 0.27151725, 0.27972708,0.27602996, 0.27893814,
                0.27644574, 0.2752384, 0.2711833, 0.27788198, 0.273968, 0.27059734, 0.277892, 0.279730407, 0.2763259,0.27228255,
                0.27030022, 0.27953874, 0.27943303, 0.2793195, 0.2789451, 0.27558493, 0.27508783, 0.27299043, 0.2727427,0.27945984,
                0.27814236, 0.27671108, 0.27544263, 0.27872882, 0.27707935, 0.27703355, 0.27446495, 0.27310665,0.2715995, 0.2789455,
                0.27860152, 0.27250125, 0.27202015, 0.27170103, 0.27163418, 0.27961522, 0.27702242, 0.27296478,0.27284566, 0.2721916]



ALSGD_cost = [1.43, 2.86, 4.29, 5.72, 7.15, 8.58, 10.01, 11.44, 12.87, 14.3,
              15.73, 17.16, 18.59, 20.02, 21.45, 22.88, 24.31, 25.75, 27.18, 28.61,
              30.04, 31.47, 32.9, 34.33, 35.76, 37.19, 38.62, 40.05, 41.48, 42.91,
              44.34, 45.77, 47.2, 48.63, 50.06, 51.49, 52.92, 54.35, 55.78, 57.21,
              58.64, 60.07, 61.5, 62.93, 64.36, 65.79, 67.22, 68.65, 70.08, 71.51,
              72.94, 74.37, 75.8, 77.24, 78.67, 80.1, 81.53, 82.96, 84.39, 85.82,
              87.25, 88.68, 90.11, 91.54, 92.97, 94.4, 95.83, 97.26, 98.69, 100.12,
              101.55, 102.98, 104.41, 105.84, 107.27, 108.7, 110.13, 111.56, 112.99, 114.42,
              115.85, 117.28, 118.71, 120.14, 121.57, 123.0, 124.43, 125.86, 127.3, 128.73,
              130.16, 131.59, 133.02, 134.45, 135.88, 137.31, 138.74, 140.17, 141.6, 143.03]

FedAvg_cost = [4.96, 9.91, 14.87, 19.83, 24.79, 29.74, 34.7, 39.66, 44.62, 49.57,
              54.53, 59.49, 64.44, 69.4, 74.36, 79.32, 84.27, 89.23, 94.19, 99.14,
              104.1, 109.06, 114.02, 118.97, 123.93, 128.89, 133.85, 138.8, 143.76, 148.72,
              153.67, 158.63, 163.59, 168.55, 173.5, 178.46, 183.42, 188.37, 193.33, 198.29,
              203.25, 208.2, 213.16, 218.12, 223.08, 228.03, 232.99, 237.95, 242.9, 247.86,
              252.82, 257.78, 262.73, 267.69, 272.65, 277.6, 282.56, 287.52, 292.48, 297.43,
              302.39, 307.35, 312.31, 317.26, 322.22, 327.18, 332.13, 337.09, 342.05, 347.01,
              351.96, 356.92, 361.88, 366.83, 371.79, 376.75, 381.71, 386.66, 391.62, 396.58,
              401.54, 406.49, 411.45, 416.41, 421.36, 426.32, 431.28, 436.24, 441.19, 446.15,
              451.11, 456.07, 461.02, 465.98, 470.94, 475.89, 480.85, 485.81, 490.77, 495.72]

FedProx_cost = [2.421, 4.842, 7.263, 9.684, 12.104999999999999, 14.526, 16.947, 19.368, 21.788999999999998, 24.209999999999997, 26.630999999999997,
                29.052, 31.473, 33.894, 36.315, 38.736, 41.157, 43.577999999999996, 45.998999999999995, 48.419999999999995, 50.840999999999994,
                53.26199999999999, 55.68299999999999, 58.104, 60.525, 62.946, 65.36699999999999, 67.788, 70.20899999999999, 72.63, 75.05099999999999,
                77.472, 79.893, 82.314, 84.735, 87.15599999999999, 89.577, 91.99799999999999, 94.419, 96.83999999999999, 99.261, 101.68199999999999,
                104.103, 106.52399999999999, 108.945, 111.36599999999999, 113.78699999999999, 116.208, 118.62899999999999, 121.05, 123.47099999999999,
                125.892, 128.313, 130.73399999999998, 133.155, 135.576, 137.99699999999999, 140.41799999999998, 142.839, 145.26, 147.68099999999998,
                150.10199999999998, 152.523, 154.944, 157.36499999999998, 159.786, 162.207, 164.628, 167.04899999999998, 169.47, 171.891, 174.31199999999998,
                176.73299999999998, 179.154, 181.575, 183.99599999999998, 186.41699999999997, 188.838, 191.259, 193.67999999999998, 196.101, 198.522,
                200.94299999999998, 203.36399999999998, 205.785, 208.206, 210.62699999999998, 213.04799999999997, 215.469, 217.89, 220.31099999999998,
                222.73199999999997, 225.153, 227.57399999999998, 229.99499999999998, 232.416, 234.837, 237.25799999999998, 239.67899999999997, 242.1]




ALSGD_acc1 = [0.8145682, 0.8225607, 0.835356394, 0.8443456935, 0.850452, 0.857562, 0.8644565, 0.8672347, 0.8705639, 0.8756327,
             0.88062393, 0.8862452573, 0.891242323, 0.8955628, 0.89865327, 0.9002345066, 0.903224855, 0.90542587, 0.9074528967, 0.90845264,

             0.90870085, 0.910408587, 0.91035803, 0.9104625, 0.9105968, 0.91069, 0.91036577,0.9105936, 0.9101996, 0.9105649,
             0.9107878, 0.9106215, 0.91080587, 0.9108383, 0.91072853, 0.91091875, 0.91082136,0.91077933, 0.91064985, 0.9106426,
             0.9106853, 0.91067145, 0.9103099, 0.9105571, 0.9107996, 0.9109508, 0.910584,0.91016457, 0.91035916, 0.91052747,
             0.91054876, 0.91066256, 0.91075766, 0.91037375, 0.9106782, 0.91039421, 0.91083144,0.910174634, 0.91030222, 0.91047373,
             0.910774976, 0.91025776, 0.9104216, 0.9106473, 0.9104132, 0.9101234, 0.91042295,0.9113151, 0.9114764, 0.9116243,
             0.9118107, 0.9117794, 0.91162244, 0.9112263, 0.9114062, 0.9117539, 0.9117333, 0.9118682, 0.9119745, 0.91178805,
             0.9119897, 0.91187176, 0.9117662, 0.9115129, 0.9120407, 0.9122666, 0.91237466, 0.91276894, 0.9127552, 0.91288884,
             0.91272765, 0.9126406, 0.9127509, 0.9129101, 0.91267383, 0.91279156, 0.9128866,0.912858,0.9129356,0.91278556]



FedProx_acc1 = [0.80012334, 0.8069462, 0.8120308, 0.81805, 0.822351705, 0.8275076, 0.834422, 0.84055736, 0.84565356, 0.850282,
              0.855142, 0.858264, 0.8622615, 0.8670326, 0.87108844, 0.87502895, 0.87906706, 0.88204174, 0.88503676, 0.88800376,
              0.8920903, 0.8950573, 0.896953, 0.89808182, 0.90006028854, 0.901030496, 0.90198331, 0.903093936, 0.904089754, 0.905061864,

              0.906063, 0.906452, 0.906562, 0.9065632, 0.9066781, 0.9066892, 0.9063783, 0.9067825, 0.9066568, 0.9065325,
              0.906051076, 0.90608487, 0.906430544, 0.9067477, 0.9069786, 0.9068413735, 0.906919504, 0.906527644, 0.906663436, 0.906773634,
              0.90684269, 0.90665854, 0.90659662, 0.906860567, 0.9061956, 0.90657333, 0.90649744, 0.9067327, 0.90697146, 0.90696734,
              0.9060884, 0.90659093883, 0.906092107, 0.90649612, 0.90669907925, 0.90659014935, 0.9066970907, 0.90699121, 0.9068018304, 0.906509049,
              0.90510551, 0.90631569, 0.9065165, 0.90634119, 0.9064672, 0.90677197, 0.90698356, 0.90638613, 0.90679405, 0.90603455,
              0.9066887, 0.9068799, 0.9067479, 0.90744265, 0.90753724, 0.9077084, 0.9077905, 0.90787955, 0.9078467, 0.9074742,
              0.9077073, 0.9080094, 0.90857417, 0.9081403, 0.90842416, 0.90817937, 0.90839365, 0.9084179, 0.90864034, 0.90834875
            ]

FedAvg_acc1 = [0.80356576, 0.80996814, 0.8158215, 0.8196135, 0.8226245, 0.826824, 0.8305775, 0.8350515, 0.839093586, 0.84344844,
               0.8484852, 0.8526525, 0.85614316, 0.8603086, 0.863225335, 0.86637, 0.86962141, 0.8717996, 0.8756633, 0.877072,
               0.8809176, 0.882120967, 0.88432135, 0.88651433, 0.8880161, 0.8900675, 0.89211904, 0.8937796, 0.8941137, 0.89509627,
               0.89608145, 0.897025148, 0.898921529, 0.9000407216, 0.9005036546, 0.9010391, 0.9016856, 0.902060569, 0.90244476, 0.90308557,

               0.90405625, 0.90408733, 0.90413763, 0.904122, 0.90414671, 0.904146509, 0.90451724, 0.90455348, 0.90456327, 0.90461886,
               0.90462956, 0.904704995, 0.90472667, 0.90465876, 0.90476055, 0.9048746, 0.90493775, 0.9048, 0.9047202, 0.9047641,
               0.9045978, 0.9046293, 0.9044915, 0.9042239, 0.9043218, 0.90475225, 0.9039214, 0.90409913, 0.90406145, 0.90404196,
               0.90425482, 0.904160564, 0.9042667, 0.904167024, 0.9040078, 0.90433709, 0.90425472, 0.904291, 0.9045599, 0.9046443,
               0.90435054, 0.9047949, 0.9048305, 0.9043339, 0.9047993, 0.9049755, 0.9052414, 0.9055823, 0.90556593, 0.905557,
               0.9055096, 0.90579034, 0.9058882, 0.90565255, 0.90526724, 0.9056996, 0.9058128, 0.9057889, 0.90549294,0.9058525 ]




ALSGD_loss1 = [0.8371139, 0.80704224,0.7717365,0.7398699, 0.6929115,0.65211835, 0.6144244, 0.5879254, 0.5567385, 0.51055515,
              0.4806134, 0.46015536, 0.44017934, 0.42819803, 0.41950018, 0.4102217, 0.4008732,0.39307517, 0.3844945, 0.3797536,

              0.36899332, 0.3662787, 0.36673068, 0.36632707, 0.36751904, 0.36641695, 0.3662981, 0.3647934, 0.36526135, 0.3657245,
              0.3665281, 0.36562243, 0.36543685, 0.36433667, 0.3631085, 0.36157552, 0.3608635, 0.3623324, 0.3616587, 0.36159404,
              0.3626541, 0.3624806, 0.36303708, 0.36204103, 0.36454196, 0.36243765, 0.36408116, 0.3665732, 0.36440222, 0.36375768,
              0.36477614, 0.36631247, 0.3662091, 0.36602458, 0.36556017, 0.36441716, 0.3642331, 0.3637303, 0.36521176, 0.3643077,
              0.36477395, 0.363764912, 0.3659643, 0.3638547, 0.3615704, 0.3604987, 0.362373, 0.3616775, 0.36342921, 0.36353598,
              0.36241967, 0.36335024, 0.3622, 0.36088224, 0.3632939, 0.3647798, 0.36247508, 0.3616546, 0.36204902, 0.3614977,
              0.36121555, 0.36339283, 0.36275447, 0.36441706, 0.3628685, 0.364902, 0.3635686, 0.3636803, 0.36240072, 0.361471498,
              0.36289758, 0.3632918, 0.36199161, 0.36344365, 0.36418105, 0.36130676, 0.36251132, 0.36079613, 0.36288516, 0.3617263]


FedProx_loss1 = [0.83880824, 0.7998019, 0.76366535, 0.735715636, 0.69031606, 0.668496, 0.64844333, 0.6237152, 0.60586807,0.58937494,
                0.5701819, 0.5552123, 0.529047036, 0.518361, 0.50608502, 0.49583045, 0.4856188, 0.47823932, 0.470215403,0.46010611,
                0.4521805, 0.4455158, 0.43831005, 0.42856266, 0.42082464, 0.4124105, 0.40739102, 0.402337738, 0.398095444, 0.393993395,

                0.38977328, 0.38870442, 0.38744146, 0.38885523, 0.3887656, 0.38885512, 0.3878665, 0.38645384,0.38611573, 0.385222,
                0.38555062, 0.38679703, 0.38603977, 0.3862474, 0.38533835, 0.38493748, 0.38558707, 0.38490233,0.3835048, 0.38490435,
                0.3848447, 0.3856367, 0.3846028, 0.38437855, 0.38545676, 0.38346522, 0.38451725, 0.38472708,0.38602996, 0.38493814,
                0.38544574, 0.3852384, 0.384833, 0.38388198, 0.383968, 0.38459734, 0.382892, 0.383730407, 0.3833259,0.3828255,
                0.38230022, 0.38253874, 0.3833303, 0.3833195, 0.3829451, 0.38258493, 0.38208783, 0.38299043, 0.3827427,0.38245984,
                0.38214236, 0.38171108, 0.38144263, 0.38172882, 0.38107935, 0.38103355, 0.38046495, 0.38110665,0.3815995, 0.3819455,
                0.38260152, 0.38150125, 0.38202015, 0.38170103, 0.38163418, 0.38161522, 0.38102242, 0.38096478,0.38084566, 0.3811916]

FedAvg_loss1 = [0.891351,0.8763474, 0.8568089, 0.79209226, 0.7543053, 0.7275452, 0.7076524, 0.6858916, 0.66077377, 0.63131656,
               0.6006413,  0.57908849, 0.564001, 0.5505933, 0.541384, 0.5301536, 0.52078167,0.5165676, 0.51205448, 0.5054464,
               0.49768977, 0.4908313, 0.48650317, 0.4794696, 0.4749926, 0.46945056, 0.46564536, 0.46190815, 0.4577835, 0.45205247,
               0.4488001, 0.44470627, 0.44065274, 0.43609032, 0.43322062, 0.43085515, 0.42707008, 0.42395874, 0.41825136, 0.412673173,

               0.40986308, 0.4090617, 0.4097533, 0.40892256, 0.40853544, 0.40793358, 0.4070152, 0.4063812, 0.40584905, 0.40609968,
               0.40597295, 0.40665743, 0.40650855, 0.40642213, 0.405919816, 0.40623273, 0.405605935, 0.40547783, 0.40604955, 0.4056584,
               0.4061802, 0.4053473, 0.40500556, 0.40523515, 0.40594314, 0.40639172, 0.4052201, 0.40479634, 0.40452677, 0.40501438,
               0.4041339, 0.4046954, 0.40450115, 0.40452327, 0.4040963, 0.40462326, 0.40388933, 0.403116493, 0.40332124, 0.4031575,
               0.40354482, 0.403078, 0.40317884, 0.402913332, 0.4027482, 0.403473592, 0.402920784, 0.402892606, 0.403418517, 0.402653788,
               0.402505932, 0.40369035, 0.402825546, 0.40298996, 0.403223818, 0.402887245, 0.402772306, 0.402625593, 0.403203172, 0.402581974]


ALSGD_acc2 = [0.8605682, 0.874607, 0.888356394, 0.8983456935, 0.906452, 0.9148862, 0.920565, 0.9252347, 0.9295639, 0.9326327,
             0.93562393, 0.9372452573, 0.939242323, 0.9415628, 0.94365327, 0.9442345066, 0.945224855, 0.94642587, 0.9474528967, 0.94845264,

             0.94920085, 0.949408587, 0.94935803, 0.9494625, 0.9495968, 0.9493, 0.9496577, 0.949936, 0.949996, 0.949649,
             0.9507878, 0.950215, 0.95080587, 0.9508383, 0.95072853, 0.9501875, 0.9502136, 0.9507933, 0.9504985, 0.950426,
             0.9516853, 0.95167145, 0.9513099, 0.9515571, 0.9517996, 0.951508, 0.95184, 0.95106457, 0.95125916, 0.95152747,
             0.95124876, 0.95136256, 0.95115766, 0.95247375, 0.9526782, 0.95239421, 0.95283144, 0.952374634, 0.95230222, 0.95247373,
             0.952774976, 0.95225776, 0.9524216, 0.9536473, 0.953132, 0.9531234, 0.95342295,0.9533151, 0.9534764, 0.953243,
             0.9535107, 0.9547794, 0.9542244, 0.9542263, 0.9544062, 0.9547539, 0.9547333, 0.9548682, 0.9547745, 0.95478805,
             0.9549897, 0.95487176, 0.9547662, 0.9545129, 0.9540407, 0.9542666, 0.95437466, 0.95476894, 0.9547552, 0.95408884,
             0.95422765, 0.9546406, 0.9547509, 0.9549101, 0.95467383, 0.95479156, 0.9548866, 0.954858, 0.9549356, 0.95478556]

FedProx_acc2 = [0.850012334, 0.8569462, 0.8610308, 0.86705, 0.872351705, 0.8785076, 0.884422, 0.88955736, 0.89465356, 0.898282,
              0.902142, 0.906264, 0.9102615, 0.9130326, 0.91708844, 0.92002895, 0.92306706, 0.92604174, 0.92903676, 0.93100376,
              0.9330903, 0.9350573, 0.937953, 0.93908182, 0.94006028854, 0.941030496, 0.94298331, 0.943093936, 0.944089754, 0.945061864,

              0.946063, 0.946452, 0.946562, 0.9465632, 0.9466781, 0.9466892, 0.9463783, 0.9467825, 0.9466568, 0.9465325,
              0.947051076, 0.94708487, 0.947430544, 0.9477477, 0.9479786, 0.9478413735, 0.947919504, 0.947527644, 0.947663436, 0.947773634,
              0.94884269, 0.94865854, 0.94859662, 0.948860567, 0.9481956, 0.94857333, 0.94849744, 0.9487327, 0.94897146, 0.94896734,
              0.9490884, 0.94959093883, 0.949092107, 0.94949612, 0.94969907925, 0.94959014935, 0.9496970907, 0.94999121, 0.9498018304, 0.949509049,
              0.94910551, 0.94931569, 0.9505165, 0.95034119, 0.9504672, 0.95077197, 0.95098356, 0.95038613, 0.95079405, 0.95003455,
              0.9506887, 0.9508799, 0.9507479, 0.95044265, 0.95053724, 0.9507084, 0.9507905, 0.95087955, 0.9508467, 0.9504742,
              0.9507073, 0.9500094, 0.95057417, 0.9501403, 0.95042416, 0.95017937, 0.95039365, 0.9504179, 0.95064034, 0.95034875
            ]

FedAvg_acc2 = [0.83056576, 0.83896814, 0.8428215, 0.8466135, 0.8516245, 0.855824, 0.8605775, 0.8650515, 0.869093586, 0.87444844,
               0.8784852, 0.8826525, 0.88614316, 0.8903086, 0.893225335, 0.89737, 0.90162141, 0.905996, 0.9096633, 0.9120072,
               0.9153176, 0.918120967, 0.92132135, 0.9241433, 0.92600161, 0.9278675, 0.92901904, 0.931010796, 0.932007137, 0.933209627,
               0.93463268145, 0.935835148, 0.93673521529, 0.93763407216, 0.93840036546, 0.9392391, 0.94000856, 0.941160569, 0.94224476, 0.94328557,

               0.943805625, 0.94408733, 0.94413763, 0.944322, 0.94424671, 0.944546509, 0.94431724, 0.94465348, 0.94486327, 0.94501886,
               0.94532956, 0.945204995, 0.94542667, 0.94565876, 0.94576055, 0.9454746, 0.94553775, 0.9458, 0.9458202, 0.9460641,
               0.9462978, 0.9461293, 0.9464915, 0.9466239, 0.9468218, 0.94655225, 0.9469214, 0.94689913, 0.94706145, 0.94744196,
               0.94725482, 0.947360564, 0.9475667, 0.947767024, 0.9479078, 0.94823709, 0.94855472, 0.948191, 0.9483599, 0.9486443,
               0.94855054, 0.9487949, 0.9488305, 0.9492339, 0.9494993, 0.9495755, 0.9493414, 0.9495823, 0.94976593, 0.949557,
               0.9496096, 0.94999034, 0.9502882, 0.95045255, 0.950326724, 0.95046996, 0.95028128, 0.95037889, 0.95049294,0.95038525 ]

ALSGD_loss2 = [0.8071139, 0.74704224,0.6917365,0.6598699, 0.6129115,0.56211835, 0.5344244, 0.4979254, 0.4667385, 0.43055515,
              0.4006134, 0.39015536, 0.37017934, 0.35819803, 0.33950018, 0.3202217, 0.3108732,0.307517, 0.2944945, 0.2887536,

              0.28459332, 0.2826787, 0.27833068, 0.27742707, 0.27511904, 0.27891695, 0.2745981, 0.2761934, 0.27896135, 0.2754245,
              0.2791281, 0.27662243, 0.27343685, 0.27873667, 0.2761085, 0.27747552, 0.274135, 0.2765324, 0.2782587, 0.27919404,
              0.276941, 0.2754806, 0.27803708, 0.27454103, 0.27514196, 0.27763765, 0.27688116, 0.2725732, 0.27530222, 0.27675768,
              0.27597614, 0.27231247, 0.2747091, 0.27522458, 0.27656017, 0.27711716, 0.2737331, 0.2754303, 0.27811176, 0.2757077,
              0.2734395, 0.274064912, 0.2769643, 0.27442547, 0.2757704, 0.2766987, 0.273373, 0.2726775, 0.27472921, 0.27153598,
              0.27421967, 0.27375024, 0.2722, 0.27488224, 0.2736939, 0.2727798, 0.27587508, 0.2746546, 0.27204902, 0.273477,
              0.27421555, 0.27119283, 0.27285447, 0.27321706, 0.2748685, 0.273902, 0.2735686, 0.2716803, 0.27280072, 0.270471498,
              0.27219758, 0.2702918, 0.27339161, 0.27434365, 0.27118105, 0.27160676, 0.27331132, 0.27339613, 0.270868516, 0.2702263]


FedProx_loss2 = [0.7880824, 0.758019, 0.7266535, 0.6915636, 0.6631606, 0.63496, 0.6044333, 0.581152, 0.5616807,0.547494,
                0.5201819, 0.502123, 0.48047036, 0.468361, 0.4508502, 0.44045, 0.43188, 0.423932, 0.410215403,0.40010611,
                0.39521805, 0.385158, 0.3731005, 0.3656266, 0.35082464, 0.3424105, 0.335102, 0.32337738, 0.31095444, 0.30003395,

                0.29477328, 0.29070442, 0.28744146, 0.28885523, 0.2827656, 0.28885512, 0.2838665, 0.28645384,0.28411573, 0.285222,
                0.28655062, 0.28379703, 0.28403977, 0.2852474, 0.28733835, 0.28493748, 0.28258707, 0.28390233,0.2825048, 0.28590435,
                0.2848447, 0.2856367, 0.2826028, 0.28137855, 0.28445676, 0.28646522, 0.28451725, 0.28572708,0.28702996, 0.28493814,
                0.28244574, 0.2862384, 0.283833, 0.28588198, 0.283968, 0.28259734, 0.285892, 0.283730407, 0.2813259,0.2858255,
                0.28230022, 0.28453874, 0.2833303, 0.2863195, 0.2839451, 0.28558493, 0.28308783, 0.28399043, 0.2857427,0.28245984,
                0.28214236, 0.28171108, 0.28444263, 0.28172882, 0.28307935, 0.28503355, 0.28146495, 0.28310665,0.2825995, 0.2809455,
                0.28260152, 0.28150125, 0.28302015, 0.28170103, 0.28363418, 0.28161522, 0.28402242, 0.28296478,0.28184566, 0.2801916]

FedAvg_loss2 = [0.931351,0.9163474, 0.8968089, 0.86209226, 0.843053, 0.825452, 0.806524, 0.780916, 0.75777377, 0.72831656,
               0.7076413,  0.6908849, 0.672001, 0.6505933, 0.631384, 0.6101536, 0.60078167,0.5815676, 0.56205448, 0.5474464,
               0.52768977, 0.5008313, 0.49150317, 0.4704696, 0.4619926, 0.45345056, 0.44264536, 0.43190815, 0.4217835, 0.41205247,
               0.4028001, 0.39170627, 0.38065274, 0.37209032, 0.36322062, 0.35115, 0.342707008, 0.3335874, 0.32325136, 0.312673173,

               0.30986308, 0.3080617, 0.3067533, 0.30892256, 0.30953544, 0.30793358, 0.3080152, 0.3063812, 0.30584905, 0.30809968,
               0.30797295, 0.30665743, 0.30850855, 0.30642213, 0.305919816, 0.30623273, 0.307605935, 0.30547783, 0.30604955, 0.3086584,
               0.3061802, 0.3053473, 0.30400556, 0.30723515, 0.30694314, 0.30539172, 0.3072201, 0.30479634, 0.30552677, 0.30601438,
               0.3041339, 0.3036954, 0.30650115, 0.30452327, 0.3050963, 0.30462326, 0.30388933, 0.304116493, 0.30532124, 0.3031575,
               0.30654482, 0.303078, 0.30417884, 0.305913332, 0.3027482, 0.303473592, 0.305920784, 0.304892606, 0.303418517, 0.301653788,
               0.303505932, 0.30469035, 0.302825546, 0.30198996, 0.303223818, 0.302887245, 0.300772306, 0.302625593, 0.301203172, 0.302581974]


def kd_show_f1(teacher_acc, student_acc, student_kd_acc):
    x = list(range(1, local_epochs + 1))

    plt.plot(x, [teacher_acc[i] for i in range(local_epochs)], label="Teacher")
    plt.plot(x, [student_acc[i] for i in range(local_epochs)], label="Student")
    plt.plot(x, [student_kd_acc[i] for i in range(local_epochs)], label="Student_kd")

    plt.xlabel('Number of iterations', fontsize=30)
    plt.ylabel('F1-score', fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(loc= 4,fontsize =24)

    plt.show()

def kd_show_loss(teacher_history, student_history, student_kd_history):
    x = list(range(1, local_epochs + 1))

    plt.plot(x, [teacher_history[i] for i in range(local_epochs)], label="Teacher")
    plt.plot(x, [student_history[i] for i in range(local_epochs)], label="Student")
    plt.plot(x, [student_kd_history[i] for i in range(local_epochs)], label="Student_kd")

    plt.xlabel('Number of iterations', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(loc='upper right',fontsize =24)

    plt.show()


def autolabel(rects, ax):
    """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=18)   # 每个bar上面的字体大小


def draw_bar_dataprocessing():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

    labels =  ["ACC", "Precision", "Recall","F1-score"]
    Smote = [97.5, 91.8, 97.5, 94.6]
    subsampled = [88.4, 91.8, 88.4, 90.1]
    sample = [99.9, 55.1, 99.9, 71.0]

    x = np.arange(len(labels))  # 标签位置

    width = 0.2  # 柱状图的宽度

    fig, ax = plt.subplots()

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    rects1 = ax.bar(x - width, Smote, width, label='IDW-Smote')
    rects2 = ax.bar(x + 0.04, subsampled, width, label='Smote')
    rects3 = ax.bar(x + width + 0.08, sample, width, label='Unprocessed data')

    # 为y轴、标题和x轴等添加一些文本。
    #ax.set_ylabel('Y轴', fontsize=16)
    #ax.set_xlabel('X轴', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize= 24)  # 横坐标 ACC各个lable的字体大小
    plt.legend(fontsize =20)  # 指示框字体大小



    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    plt.yticks(fontsize=24)  # 纵坐标 百分比的字体大小
    plt.ylim(ymin=40,ymax=115)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))  # #把y轴的主刻度设置为4的倍数
    plt.xlabel('Evaluation Metrics', fontsize=30)
    plt.ylabel('Score(%)', fontsize=30)

    fig.tight_layout()
    plt.show()

def bar_authecoder():
    x = ["Distillation", "Decision \nTree", "Random \nForest", "Logistic \nRegression", "SVM", "KNN"]
    bar_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:gray','tab:brown']
    y = [94.6, 81.1, 73.5, 76.57, 77.7, 91.3]
    # 正确显示中文和负号
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 画图，plt.bar()可以画柱状图
    rect1 = plt.bar(x, y,color=bar_colors, linewidth=0.7)
    plt.bar_label(rect1, fontsize = 18)


    plt.ylim(ymin=50)
    plt.yticks(fontsize=24)  # 纵坐标 百分比的字体大小
    plt.xticks(fontsize=24)

    # 设置y轴标签名
    plt.ylabel('F1-score(%)',fontsize=30)
    # 显示
    plt.show()



def show_aggregation(ALSGD, FedProx, FedAvg,title, loc):
    x = list(range(1, aggregate_epoch + 1))

    plt.plot(x, [ALSGD[i] for i in range(aggregate_epoch)], label="Aperiodic SGD")
    plt.plot(x, [FedAvg[i] for i in range(aggregate_epoch)], label="FedAvg")
    plt.plot(x, [FedProx[i] for i in range(aggregate_epoch)], label="FedProx")


    plt.xlabel('Communication rounds', fontsize=30)
    plt.ylabel(title, fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(loc=loc, fontsize=24)
     plt.show()

if __name__ == '__main__':
    # draw_bar_dataprocessing()
    # bar_authecoder()
    #show_aggregation(ALSGD_acc, FedProx_acc, FedAvg_acc,'F1-score',4)
    #show_aggregation(ALSGD_loss, FedProx_loss, FedAvg_loss, 'Loss',1)
    # show_aggregation(ALSGD_cost, FedProx_cost, FedAvg_cost, 'Communication Cost (GiB)', 2)
    #kd_show_f1(teacher_acc, student_acc, student_kd_acc)
    #kd_show_loss(teacher_history, student_history, student_kd_history)

    # show_aggregation(ALSGD_acc1, FedProx_acc1, FedAvg_acc1,'F1-score',4)
    # show_aggregation(ALSGD_loss1, FedProx_loss1, FedAvg_loss1, 'Loss',1)

    #show_aggregation(ALSGD_acc2, FedProx_acc2, FedAvg_acc2,'F1-score',4)
    show_aggregation(ALSGD_loss2, FedProx_loss2, FedAvg_loss2, 'Loss',1)