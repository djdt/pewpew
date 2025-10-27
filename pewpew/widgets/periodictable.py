import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.objects import KeepMenuOpenFilter

isotope_data = np.array(
    [
        ("H", 1, 1.00782501698, 0.99988502264, 0),
        ("H", 2, 2.0141017437, 0.000115000002552, 0),
        ("He", 3, 3.01602935791, 1.34000003982e-06, 0),
        ("He", 4, 4.00260305405, 0.999998688698, 0),
        ("Li", 6, 6.01512289047, 0.075900003314, 0),
        ("Li", 7, 7.0160036087, 0.924099981785, 1),
        ("Be", 9, 9.01218318939, 1, 1),
        ("B", 10, 10.0129365921, 0.199000000954, 0),
        ("B", 11, 11.0093050003, 0.800999999046, 1),
        ("C", 12, 12, 0.989300012589, 0),
        ("C", 13, 13.0033550262, 0.01070000045, 0),
        ("N", 14, 14.0030736923, 0.996360003948, 0),
        ("N", 15, 15.0001087189, 0.00364000000991, 0),
        ("O", 16, 15.9949150085, 0.997569978237, 0),
        ("O", 17, 16.9991321564, 0.000380000012228, 0),
        ("O", 18, 17.9991588593, 0.00205000001006, 0),
        ("F", 19, 18.9984035492, 1, 0),
        ("Ne", 20, 19.99243927, 0.904799997807, 0),
        ("Ne", 21, 20.9938468933, 0.00270000007004, 0),
        ("Ne", 22, 21.9913845062, 0.0925000011921, 0),
        ("Na", 23, 22.9897689819, 1, 0),
        ("Mg", 24, 23.985042572, 0.789900004864, 1),
        ("Mg", 25, 24.9858360291, 0.10000000149, 0),
        ("Mg", 26, 25.9825935364, 0.110100001097, 0),
        ("Al", 27, 26.9815387726, 1, 1),
        ("Si", 28, 27.9769268036, 0.922230005264, 1),
        ("Si", 29, 28.9764938354, 0.0468499995768, 0),
        ("Si", 30, 29.9737701416, 0.0309200007468, 0),
        ("P", 31, 30.9737625122, 1, 1),
        ("S", 32, 31.972070694, 0.949899971485, 1),
        ("S", 33, 32.9714584351, 0.00749999983236, 0),
        ("S", 34, 33.9678688049, 0.042500000447, 0),
        ("S", 36, 35.9670791626, 9.99999974738e-05, 0),
        ("Cl", 35, 34.9688529968, 0.757600009441, 1),
        ("Cl", 37, 36.9659042358, 0.24240000546, 0),
        ("Ar", 36, 35.9675445557, 0.00333600002341, 0),
        ("Ar", 38, 37.9627304077, 0.000629000016488, 0),
        ("Ar", 40, 39.9623832703, 0.99603497982, 0),
        ("K", 39, 38.9637069702, 0.932581007481, 1),
        ("K", 40, 39.9639968872, 0.000117000003229, 0),
        ("K", 41, 40.9618263245, 0.0673020035028, 0),
        ("Ca", 40, 39.9625892639, 0.969410002232, 0),
        ("Ca", 42, 41.9586181641, 0.00646999990568, 0),
        ("Ca", 43, 42.9587669373, 0.00135000003502, 1),
        ("Ca", 44, 43.9554824829, 0.0208599995822, 1),
        ("Ca", 46, 45.9536895752, 3.99999989895e-05, 0),
        ("Ca", 48, 47.9525222778, 0.00187000003643, 0),
        ("Sc", 45, 44.955909729, 1, 1),
        ("Ti", 46, 45.9526290894, 0.0825000032783, 0),
        ("Ti", 47, 46.9517593384, 0.0744000002742, 1),
        ("Ti", 48, 47.9479408264, 0.737200021744, 0),
        ("Ti", 49, 48.9478645325, 0.0540999993682, 0),
        ("Ti", 50, 49.9447860718, 0.0518000014126, 0),
        ("V", 50, 49.9471549988, 0.00249999994412, 0),
        ("V", 51, 50.9439582825, 0.997500002384, 1),
        ("Cr", 50, 49.9460411072, 0.0434500016272, 0),
        ("Cr", 52, 51.9405059814, 0.837890028954, 1),
        ("Cr", 53, 52.9406471252, 0.0950099974871, 0),
        ("Cr", 54, 53.9388809204, 0.0236499998719, 0),
        ("Mn", 55, 54.9380455017, 1, 1),
        ("Fe", 54, 53.9396095276, 0.058449998498, 0),
        ("Fe", 56, 55.9349365234, 0.91754001379, 1),
        ("Fe", 57, 56.9353942871, 0.021190000698, 0),
        ("Fe", 58, 57.9332733154, 0.00282000005245, 0),
        ("Co", 59, 58.9331932068, 1, 1),
        ("Ni", 58, 57.9353408813, 0.680769979954, 0),
        ("Ni", 60, 59.9307861328, 0.262230008841, 1),
        ("Ni", 61, 60.9310569763, 0.0113989999518, 0),
        ("Ni", 62, 61.9283447266, 0.0363459996879, 0),
        ("Ni", 64, 63.9279670715, 0.00925500039011, 0),
        ("Cu", 63, 62.9295959473, 0.691500008106, 1),
        ("Cu", 65, 64.9277877808, 0.308499991894, 0),
        ("Zn", 64, 63.9291419983, 0.49169999361, 0),
        ("Zn", 66, 65.92603302, 0.277300000191, 1),
        ("Zn", 67, 66.9271240234, 0.0403999984264, 0),
        ("Zn", 68, 67.9248428345, 0.18449999392, 0),
        ("Zn", 70, 69.9253158569, 0.00609999988228, 0),
        ("Ga", 69, 68.9255752563, 0.601080000401, 0),
        ("Ga", 71, 70.9247055054, 0.398919999599, 1),
        ("Ge", 70, 69.9242477417, 0.20569999516, 0),
        ("Ge", 72, 71.9220733643, 0.274500012398, 1),
        ("Ge", 73, 72.9234619141, 0.077500000596, 0),
        ("Ge", 74, 73.9211807251, 0.365000009537, 0),
        ("Ge", 76, 75.9214019775, 0.0772999972105, 0),
        ("As", 75, 74.9215927124, 1, 1),
        ("Se", 74, 73.9224777222, 0.00889999978244, 0),
        ("Se", 76, 75.9192123413, 0.0936999991536, 0),
        ("Se", 77, 76.9199142456, 0.0763000026345, 0),
        ("Se", 78, 77.9173126221, 0.237700000405, 1),
        ("Se", 80, 79.916519165, 0.496100008488, 0),
        ("Se", 82, 81.9167022705, 0.0873000025749, 0),
        ("Br", 79, 78.9183349609, 0.506900012493, 1),
        ("Br", 81, 80.9162902832, 0.493099987507, 0),
        ("Kr", 78, 77.9203643799, 0.0035500000231, 0),
        ("Kr", 80, 79.9163818359, 0.02285999991, 0),
        ("Kr", 82, 81.913482666, 0.115929998457, 0),
        ("Kr", 83, 82.9141235352, 0.115000002086, 0),
        ("Kr", 84, 83.9114990234, 0.569869995117, 0),
        ("Kr", 86, 85.9106140137, 0.172790005803, 0),
        ("Rb", 85, 84.9117889404, 0.721700012684, 1),
        ("Rb", 87, 86.9091796875, 0.278299987316, 0),
        ("Sr", 84, 83.9134216309, 0.00559999980032, 0),
        ("Sr", 86, 85.9092636108, 0.0986000001431, 0),
        ("Sr", 87, 86.9088745117, 0.070000000298, 0),
        ("Sr", 88, 87.9056091309, 0.825800001621, 1),
        ("Y", 89, 88.9058380127, 1, 1),
        ("Zr", 90, 89.9047012329, 0.514500021935, 1),
        ("Zr", 91, 90.9056396484, 0.112199999392, 0),
        ("Zr", 92, 91.9050369263, 0.171499997377, 0),
        ("Zr", 94, 93.9063110352, 0.173800006509, 0),
        ("Zr", 96, 95.9082717896, 0.0280000008643, 0),
        ("Nb", 93, 92.9063720703, 1, 1),
        ("Mo", 92, 91.9068069458, 0.145300000906, 0),
        ("Mo", 94, 93.9050827026, 0.0914999991655, 0),
        ("Mo", 95, 94.9058380127, 0.158399999142, 1),
        ("Mo", 96, 95.9046783447, 0.166700005531, 0),
        ("Mo", 97, 96.9060211182, 0.0960000008345, 0),
        ("Mo", 98, 97.9054031372, 0.243900001049, 0),
        ("Mo", 100, 99.9074707031, 0.0982000008225, 0),
        ("Ru", 96, 95.9075927734, 0.0553999990225, 0),
        ("Ru", 98, 97.9052886963, 0.0186999998987, 0),
        ("Ru", 99, 98.9059371948, 0.127599999309, 0),
        ("Ru", 100, 99.9042129517, 0.126000002027, 0),
        ("Ru", 101, 100.905578613, 0.170599997044, 1),
        ("Ru", 102, 101.904342651, 0.315499991179, 0),
        ("Ru", 104, 103.905426025, 0.186199992895, 0),
        ("Rh", 103, 102.90549469, 1, 1),
        ("Pd", 102, 101.905601501, 0.0102000003681, 0),
        ("Pd", 104, 103.904029846, 0.111400000751, 0),
        ("Pd", 105, 104.905082703, 0.223299995065, 1),
        ("Pd", 106, 105.90348053, 0.273299992085, 0),
        ("Pd", 108, 107.903892517, 0.264600008726, 0),
        ("Pd", 110, 109.905174255, 0.117200002074, 0),
        ("Ag", 107, 106.905090332, 0.518389999866, 1),
        ("Ag", 109, 108.904754639, 0.481610000134, 0),
        ("Cd", 106, 105.906463623, 0.0125000001863, 0),
        ("Cd", 108, 107.904182434, 0.00889999978244, 0),
        ("Cd", 110, 109.903007507, 0.124899998307, 0),
        ("Cd", 111, 110.904182434, 0.12800000608, 1),
        ("Cd", 112, 111.902763367, 0.24130000174, 0),
        ("Cd", 113, 112.904411316, 0.122199997306, 0),
        ("Cd", 114, 113.903366089, 0.287299990654, 0),
        ("Cd", 116, 115.904762268, 0.0749000012875, 0),
        ("In", 113, 112.904060364, 0.0428999997675, 0),
        ("In", 115, 114.903877258, 0.957099974155, 1),
        ("Sn", 112, 111.904823303, 0.0097000002861, 0),
        ("Sn", 114, 113.902786255, 0.00659999996424, 0),
        ("Sn", 115, 114.903343201, 0.00340000004508, 0),
        ("Sn", 116, 115.901741028, 0.145400002599, 0),
        ("Sn", 117, 116.902954102, 0.0768000036478, 0),
        ("Sn", 118, 117.901603699, 0.242200002074, 1),
        ("Sn", 119, 118.903312683, 0.0859000012279, 0),
        ("Sn", 120, 119.902198792, 0.325800001621, 0),
        ("Sn", 122, 121.903442383, 0.0463000014424, 0),
        ("Sn", 124, 123.905273438, 0.0579000003636, 0),
        ("Sb", 121, 120.903808594, 0.572099983692, 1),
        ("Sb", 123, 122.904212952, 0.427899986506, 0),
        ("Te", 120, 119.904060364, 0.00089999998454, 0),
        ("Te", 122, 121.903045654, 0.0254999995232, 0),
        ("Te", 123, 122.904266357, 0.00889999978244, 0),
        ("Te", 124, 123.902816772, 0.0474000014365, 0),
        ("Te", 125, 124.904426575, 0.0706999972463, 1),
        ("Te", 126, 125.903312683, 0.188400000334, 0),
        ("Te", 128, 127.904464722, 0.31740000844, 0),
        ("Te", 130, 129.906219482, 0.340799987316, 0),
        ("I", 127, 126.904472351, 1, 1),
        ("Xe", 124, 123.905891418, 0.00095199997304, 0),
        ("Xe", 126, 125.904296875, 0.00089000002481, 0),
        ("Xe", 128, 127.903533936, 0.0191019997001, 0),
        ("Xe", 129, 128.904785156, 0.264005988836, 0),
        ("Xe", 130, 129.903503418, 0.0407099984586, 0),
        ("Xe", 131, 130.905090332, 0.212323993444, 0),
        ("Xe", 132, 131.904159546, 0.269086003304, 0),
        ("Xe", 134, 133.905395508, 0.104356996715, 0),
        ("Xe", 136, 135.907211304, 0.0885730013251, 0),
        ("Cs", 133, 132.905456543, 1, 1),
        ("Ba", 130, 129.906326294, 0.00106000003871, 0),
        ("Ba", 132, 131.905059814, 0.00101000000723, 0),
        ("Ba", 134, 133.904510498, 0.0241700001061, 0),
        ("Ba", 135, 134.905685425, 0.0659200027585, 0),
        ("Ba", 136, 135.904571533, 0.0785399973392, 0),
        ("Ba", 137, 136.905822754, 0.112319998443, 0),
        ("Ba", 138, 137.90524292, 0.716979980469, 1),
        ("La", 138, 137.907119751, 0.000888100010343, 0),
        ("La", 139, 138.906356812, 0.999111890793, 1),
        ("Ce", 136, 135.90713501, 0.00185000000056, 0),
        ("Ce", 138, 137.905990601, 0.00251000002027, 0),
        ("Ce", 140, 139.905441284, 0.884500026703, 1),
        ("Ce", 142, 141.909255981, 0.11113999784, 0),
        ("Pr", 141, 140.907653809, 1, 1),
        ("Nd", 142, 141.907730103, 0.271519988775, 0),
        ("Nd", 143, 142.909820557, 0.12173999846, 0),
        ("Nd", 144, 143.910095215, 0.237979993224, 0),
        ("Nd", 145, 144.912582397, 0.0829299986362, 0),
        ("Nd", 146, 145.913116455, 0.171890005469, 1),
        ("Nd", 148, 147.916900635, 0.0575600005686, 0),
        ("Nd", 150, 149.920898438, 0.0563799999654, 0),
        ("Sm", 144, 143.912002563, 0.030700000003, 0),
        ("Sm", 147, 146.914901733, 0.149900004268, 1),
        ("Sm", 148, 147.914825439, 0.112400002778, 0),
        ("Sm", 149, 148.917190552, 0.138199999928, 0),
        ("Sm", 150, 149.917282104, 0.0737999975681, 0),
        ("Sm", 152, 151.91973877, 0.267500013113, 0),
        ("Sm", 154, 153.922210693, 0.227500006557, 0),
        ("Eu", 151, 150.91986084, 0.478100001812, 1),
        ("Eu", 153, 152.921234131, 0.521899998188, 1),
        ("Gd", 152, 151.919799805, 0.00200000009499, 0),
        ("Gd", 154, 153.92086792, 0.0218000002205, 0),
        ("Gd", 155, 154.922637939, 0.148000001907, 0),
        ("Gd", 156, 155.922134399, 0.204699993134, 0),
        ("Gd", 157, 156.923965454, 0.156499996781, 1),
        ("Gd", 158, 157.924118042, 0.248400002718, 0),
        ("Gd", 160, 159.927062988, 0.218600004911, 0),
        ("Tb", 159, 158.925354004, 1, 1),
        ("Dy", 156, 155.924285889, 0.000560000014957, 0),
        ("Dy", 158, 157.924423218, 0.000950000016019, 0),
        ("Dy", 160, 159.925201416, 0.0232900008559, 0),
        ("Dy", 161, 160.926940918, 0.188889995217, 0),
        ("Dy", 162, 161.926803589, 0.254750013351, 0),
        ("Dy", 163, 162.928741455, 0.248960003257, 1),
        ("Dy", 164, 163.92918396, 0.2825999856, 0),
        ("Ho", 165, 164.930328369, 1, 1),
        ("Er", 162, 161.928787231, 0.00138999999035, 0),
        ("Er", 164, 163.929214478, 0.0160099994391, 0),
        ("Er", 166, 165.930297852, 0.335029989481, 1),
        ("Er", 167, 166.932052612, 0.228689998388, 0),
        ("Er", 168, 167.932373047, 0.269780009985, 0),
        ("Er", 170, 169.935470581, 0.149100005627, 0),
        ("Tm", 169, 168.93421936, 1, 1),
        ("Yb", 168, 167.933883667, 0.0012300000526, 0),
        ("Yb", 170, 169.934768677, 0.0298200007528, 0),
        ("Yb", 171, 170.936325073, 0.14090000093, 0),
        ("Yb", 172, 171.936386108, 0.216800004244, 1),
        ("Yb", 173, 172.938217163, 0.161029994488, 0),
        ("Yb", 174, 173.938873291, 0.320259988308, 0),
        ("Yb", 176, 175.942581177, 0.129960000515, 0),
        ("Lu", 175, 174.94078064, 0.974009990692, 1),
        ("Lu", 176, 175.942687988, 0.0259899999946, 0),
        ("Hf", 174, 173.940048218, 0.00159999995958, 0),
        ("Hf", 176, 175.94140625, 0.0526000000536, 0),
        ("Hf", 177, 176.943222046, 0.186000004411, 0),
        ("Hf", 178, 177.943710327, 0.272799998522, 1),
        ("Hf", 179, 178.94581604, 0.136199995875, 0),
        ("Hf", 180, 179.946563721, 0.350800007582, 0),
        ("Ta", 180, 179.947463989, 0.00012009999773, 0),
        ("Ta", 181, 180.947998047, 0.999879896641, 1),
        ("W", 180, 179.946716309, 0.001200000057, 0),
        ("W", 182, 181.948196411, 0.264999985695, 1),
        ("W", 183, 182.95022583, 0.143099993467, 0),
        ("W", 184, 183.950927734, 0.306400001049, 0),
        ("W", 186, 185.954360962, 0.284299999475, 0),
        ("Re", 185, 184.952957153, 0.374000012875, 1),
        ("Re", 187, 186.955749512, 0.625999987125, 0),
        ("Os", 184, 183.952484131, 0.000199999994948, 0),
        ("Os", 186, 185.953842163, 0.0159000009298, 0),
        ("Os", 187, 186.955749512, 0.0196000002325, 0),
        ("Os", 188, 187.955841064, 0.132400006056, 0),
        ("Os", 189, 188.958145142, 0.161500006914, 1),
        ("Os", 190, 189.958450317, 0.262600004673, 0),
        ("Os", 192, 191.961471558, 0.407799988985, 0),
        ("Ir", 191, 190.960586548, 0.372999995947, 0),
        ("Ir", 193, 192.962921143, 0.626999974251, 1),
        ("Pt", 190, 189.95993042, 0.000119999996969, 0),
        ("Pt", 192, 191.961044312, 0.00781999994069, 0),
        ("Pt", 194, 193.962677002, 0.328599989414, 0),
        ("Pt", 195, 194.964797974, 0.337799996138, 1),
        ("Pt", 196, 195.964950562, 0.252099990845, 0),
        ("Pt", 198, 197.967895508, 0.0735599994659, 0),
        ("Au", 197, 196.966567993, 1, 1),
        ("Hg", 196, 195.965835571, 0.00150000001304, 0),
        ("Hg", 198, 197.966766357, 0.0996999964118, 0),
        ("Hg", 199, 198.968276978, 0.168699994683, 0),
        ("Hg", 200, 199.968322754, 0.231000006199, 0),
        ("Hg", 201, 200.970306396, 0.131799995899, 1),
        ("Hg", 202, 201.97064209, 0.298599988222, 0),
        ("Hg", 204, 203.973495483, 0.0687000006437, 0),
        ("Tl", 203, 202.972351074, 0.295199990273, 0),
        ("Tl", 205, 204.97442627, 0.704800009727, 1),
        ("Pb", 204, 203.97303772, 0.0140000004321, 0),
        ("Pb", 206, 205.974472046, 0.240999996662, 0),
        ("Pb", 207, 206.975891113, 0.221000000834, 0),
        ("Pb", 208, 207.976654053, 0.523999989033, 1),
        ("Bi", 209, 208.980392456, 1, 1),
        ("Th", 232, 232.03805542, 1, 1),
        ("Pa", 231, 231.035888672, 1, 0),
        ("U", 234, 234.04095459, 5.4000000091e-05, 0),
        ("U", 235, 235.043930054, 0.00720399990678, 0),
        ("U", 238, 238.05078125, 0.99274200201, 1),
    ],
    dtype=[
        ("symbol", "U2"),
        ("isotope", int),
        ("mass", float),
        ("composition", float),
        ("preferred", int),
    ],
)

element_positions = {
    "H": ("Hydrogen", 0, 0),
    "He": ("Helium", 0, 17),
    "Li": ("Lithium", 1, 0),
    "Be": ("Beryllium", 1, 1),
    "B": ("Boron", 1, 12),
    "C": ("Carbon", 1, 13),
    "N": ("Nitrogen", 1, 14),
    "O": ("Oxygen", 1, 15),
    "F": ("Fluorine", 1, 16),
    "Ne": ("Neon", 1, 17),
    "Na": ("Sodium", 2, 0),
    "Mg": ("Magnesium", 2, 1),
    "Al": ("Aluminium", 2, 12),
    "Si": ("Silicon", 2, 13),
    "P": ("Phosphorus", 2, 14),
    "S": ("Suplhur", 2, 15),
    "Cl": ("Chlorine", 2, 16),
    "Ar": ("Argon", 2, 17),
    "K": ("Potassium", 3, 0),
    "Ca": ("Calcium", 3, 1),
    "Sc": ("Scandium", 3, 2),
    "Ti": ("Titanium", 3, 3),
    "V": ("Vanadium", 3, 4),
    "Cr": ("Chromium", 3, 5),
    "Mn": ("Manganese", 3, 6),
    "Fe": ("Iron", 3, 7),
    "Co": ("Cobalt", 3, 8),
    "Ni": ("Nickle", 3, 9),
    "Cu": ("Copper", 3, 10),
    "Zn": ("Zinc", 3, 11),
    "Ga": ("Gallium", 3, 12),
    "Ge": ("Germanium", 3, 13),
    "As": ("Aresenic", 3, 14),
    "Se": ("Selenium", 3, 15),
    "Br": ("Bromine", 3, 16),
    "Kr": ("Krypton", 3, 17),
    "Rb": ("Rubidium", 4, 0),
    "Sr": ("Strontium", 4, 1),
    "Y": ("Yttrium", 4, 2),
    "Zr": ("Zirconium", 4, 3),
    "Nb": ("Noibium", 4, 4),
    "Mo": ("Molybdenum", 4, 5),
    "Tc": ("Technetium", 4, 6),
    "Ru": ("Ruthenium", 4, 7),
    "Rh": ("Rhodium", 4, 8),
    "Pd": ("Paladaium", 4, 9),
    "Ag": ("Silver", 4, 10),
    "Cd": ("Cadmium", 4, 11),
    "In": ("Indium", 4, 12),
    "Sn": ("Tin", 4, 13),
    "Sb": ("Antimony", 4, 14),
    "Te": ("Tellurium", 4, 15),
    "I": ("Iodine", 4, 16),
    "Xe": ("Xenon", 4, 17),
    "Cs": ("Caesium", 5, 0),
    "Ba": ("Barium", 5, 1),
    "Hf": ("Hafnium", 5, 3),
    "Ta": ("Tantalum", 5, 4),
    "W": ("Tungsten", 5, 5),
    "Re": ("Rhenium", 5, 6),
    "Os": ("Osmium", 5, 7),
    "Ir": ("Iridium", 5, 8),
    "Pt": ("Platinum", 5, 9),
    "Au": ("Gold", 5, 10),
    "Hg": ("Mercury", 5, 11),
    "Tl": ("Thallium", 5, 12),
    "Pb": ("Lead", 5, 13),
    "Bi": ("Bismuth", 5, 14),
    "Po": ("Polonium", 5, 15),
    "At": ("Astatine", 5, 16),
    "Rn": ("Radon", 5, 17),
    "Fr": ("Francium", 6, 0),
    "Ra": ("Radium", 6, 1),
    "Rf": ("Rutherfordium", 6, 3),
    "Db": ("Dubnium", 6, 4),
    "Sg": ("Seaborgium", 6, 5),
    "Bh": ("Bohrium", 6, 6),
    "Hs": ("Hassium", 6, 7),
    "La": ("Lanthanum", 7, 2),
    "Ce": ("Cerium", 7, 3),
    "Pr": ("Praseodymium", 7, 4),
    "Nd": ("Neodymium", 7, 5),
    "Pm": ("Promethium", 7, 6),
    "Sm": ("Samarium", 7, 7),
    "Eu": ("Europium", 7, 8),
    "Gd": ("Gadolinium", 7, 9),
    "Tb": ("Terbium", 7, 10),
    "Dy": ("Dysprosium", 7, 11),
    "Ho": ("Holmium", 7, 12),
    "Er": ("Erbium", 7, 13),
    "Tm": ("Thulium", 7, 14),
    "Yb": ("Ytterbium", 7, 15),
    "Lu": ("Lutetium", 7, 16),
    "Ac": ("Actinium", 8, 2),
    "Th": ("Thorium", 8, 3),
    "Pa": ("Protactinium", 8, 4),
    "U": ("Uranium", 8, 5),
    "Np": ("Neptunium", 8, 6),
    "Pu": ("Plutonium", 8, 7),
    "Am": ("Americium", 8, 8),
    "Cm": ("Curium", 8, 9),
    "Bk": ("Berkelium", 8, 10),
    "Cf": ("Californium", 8, 11),
    "Es": ("Einsteinium", 8, 12),
    "Fm": ("Fermium", 8, 13),
    "Md": ("Mendelevium", 8, 14),
    "No": ("Nobelium", 8, 15),
    "Lr": ("Lawrencium", 8, 16),
}


class PeriodicTableButton(QtWidgets.QToolButton):
    isotopesChanged = QtCore.Signal()

    def __init__(
        self,
        isotopes: np.ndarray,
        enabled: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.isotopes = isotopes
        self.symbol = isotopes["symbol"][0]
        self.name = element_positions[self.symbol][0]
        self.number = isotopes["isotope"][0]

        self.indicator: QtGui.QColor | None = None

        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.DelayedPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setMinimumSize(QtCore.QSize(45, 45))

        self.action = QtGui.QAction(self.symbol, parent=self)
        self.action.setToolTip(self.name)
        self.action.setCheckable(True)
        self.setDefaultAction(self.action)

        self.isotope_actions = {
            iso["isotope"]: self.createAction(iso) for iso in self.isotopes
        }
        if enabled is not None:
            self.setEnabledIsotopes(enabled)

        isotopes_menu = QtWidgets.QMenu("Isotopes", parent=self)
        isotopes_menu.addActions(list(self.isotope_actions.values()))

        isotopes_menu.installEventFilter(KeepMenuOpenFilter(isotopes_menu))

        self.setMenu(isotopes_menu)
        self.setEnabled(
            any(action.isEnabled() for action in self.isotope_actions.values())
        )

        self.clicked.connect(self.selectPreferredIsotopes)
        self.isotopesChanged.connect(self.updateChecked)

    def preferred(self) -> np.ndarray:
        enabled = self.enabledIsotopes()
        pref = enabled["preferred"] > 0
        if not np.any(pref):
            if np.all(np.isnan(enabled["composition"])):
                return enabled[0]
            else:
                return enabled[np.nanargmax(enabled["composition"])]
        return enabled[pref]

    def createAction(self, isotope: np.ndarray) -> QtGui.QAction:
        text = f"{isotope['isotope']:3}: {isotope['mass']:.4f}"
        if not np.isnan(isotope["composition"]):
            text += f"\t{isotope['composition'] * 100.0:.2f}%"

        action = QtGui.QAction(text, parent=self)
        action.setCheckable(True)
        action.toggled.connect(self.isotopesChanged)
        return action

    def enabledIsotopes(self) -> np.ndarray:
        nums = np.array(
            [n for n, action in self.isotope_actions.items() if action.isEnabled()]
        )
        return self.isotopes[np.isin(self.isotopes["isotope"], nums)]

    def setEnabledIsotopes(self, enabled: np.ndarray):
        for isotope, action in self.isotope_actions.items():
            if isotope in enabled["Isotope"]:
                action.setEnabled(True)
            else:
                action.setEnabled(False)
                action.setChecked(False)

    def selectedIsotopes(self) -> np.ndarray:
        nums = np.array(
            [n for n, action in self.isotope_actions.items() if action.isChecked()]
        )
        return self.isotopes[np.isin(self.isotopes["isotope"], nums)]

    def selectPreferredIsotopes(self, checked: bool) -> None:
        preferred = self.preferred()
        for num, action in self.isotope_actions.items():
            if checked and np.isin(num, preferred["isotope"]):
                action.setChecked(True)
            else:
                action.setChecked(False)

        self.update()

    def updateChecked(self) -> None:
        if len(self.selectedIsotopes()) > 0:
            self.setChecked(True)
        else:
            self.setChecked(False)
        self.update()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        self.showMenu()  # pragma: no cover

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(option)

        font = self.font()
        font.setPointSizeF(font.pointSizeF() * 0.66)
        painter.setFont(font)

        # Draw element number
        self.style().drawItemText(
            painter,
            option.rect.adjusted(2, 0, 0, 0),  # type: ignore
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            self.palette(),
            self.isEnabled(),
            str(self.number),
        )

        # Draw number selected
        num = len(self.selectedIsotopes())
        if num > 0:
            self.style().drawItemText(
                painter,
                option.rect.adjusted(2, 0, 0, 0),  # type: ignore
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom,
                self.palette(),
                self.isEnabled(),
                f"{num}/{len(self.isotopes)}",
            )

        # Draw color icon
        if self.indicator is not None:
            rect = QtCore.QRectF(0.0, 0.0, 10.0, 10.0)
            rect.moveTopRight(option.rect.topRight() + QtCore.QPoint(-2, 3))  # type: ignore
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setBrush(QtGui.QBrush(self.indicator))
            painter.drawEllipse(rect)


class PeriodicTableSelector(QtWidgets.QWidget):
    isotopesChanged = QtCore.Signal()

    def __init__(
        self,
        enabled_isotopes: np.ndarray | None = None,
        selected_isotopes: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.pkeys: list[int] = []

        if enabled_isotopes is None:
            enabled_isotopes = isotope_data

        self.buttons: dict[str, PeriodicTableButton] = {}
        layout = QtWidgets.QGridLayout()
        row = 0
        for symbol, (_, row, col) in element_positions.items():
            # Limit to chosen ones
            isotopes = isotope_data[isotope_data["symbol"] == symbol]
            if isotopes.size > 0:
                self.buttons[symbol] = PeriodicTableButton(isotopes)
                self.buttons[symbol].isotopesChanged.connect(self.isotopesChanged)
                layout.addWidget(self.buttons[symbol], row, col)

        layout.setRowStretch(row + 1, 1)  # Last row stretch

        self.buttons["Mn"].clicked.connect(self.animate)
        self.isotopesChanged.connect(self.findCollisions)

        self.setLayout(layout)

    def animate(self):
        geom = self.buttons["Mn"].geometry()
        new_geom = geom.adjusted(-5, -5, 5, 5)

        out = QtCore.QPropertyAnimation(self.buttons["Mn"], b"geometry", self)
        out.setDuration(200)
        out.setStartValue(geom)
        out.setEndValue(new_geom)
        out.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)

        back = QtCore.QPropertyAnimation(self.buttons["Mn"], b"geometry", self)
        back.setDuration(200)
        back.setStartValue(new_geom)
        back.setEndValue(geom)
        back.setEasingCurve(QtCore.QEasingCurve.Type.OutBounce)

        anim = QtCore.QSequentialAnimationGroup(self)
        anim.addAnimation(out)
        anim.addAnimation(back)
        anim.start()

    def enabledIsotopes(self) -> np.ndarray:
        enabled: list[np.ndarray] = []
        for button in self.buttons.values():
            enabled.extend(button.enabledIsotopes())
        return np.stack(enabled)

    def setEnabledIsotopes(self, enabled: np.ndarray) -> None:
        for symbol, button in self.buttons.items():
            isotopes = enabled[enabled["symbol"] == symbol]
            button.setEnabled(isotopes.size > 0)
            button.setEnabledIsotopes(isotopes)

    def selectedIsotopes(self) -> np.ndarray | None:
        selected: list[np.ndarray] = []
        for button in self.buttons.values():
            selected.extend(button.selectedIsotopes())
        if len(selected) == 0:
            return None
        return np.stack(selected)

    def setSelectedIsotopes(self, isotopes: np.ndarray | None) -> None:
        self.blockSignals(True)
        for button in self.buttons.values():
            for action in button.isotope_actions.values():
                action.setChecked(False)
        if isotopes is not None:
            for isotope in isotopes:
                self.buttons[isotope["symbol"]].isotope_actions[
                    isotope["isotope"]
                ].setChecked(True)
        self.blockSignals(False)
        self.isotopesChanged.emit()

    def setIsotopeColors(
        self, isotopes: np.ndarray, colors: list[QtGui.QColor]
    ) -> None:
        """Set the indicator colors for ``isotopes`` to ``colors.

        Will change text to BrightText ColorRole if a dark color is used.
        Sets other buttons to the default color.
        """
        for button in self.buttons.values():
            button.indicator = None

        for isotope, color in zip(isotopes, colors):
            self.buttons[isotope["symbol"]].indicator = color

    def findCollisions(self) -> None:
        selected = self.selectedIsotopes()

        for symbol, button in self.buttons.items():
            if selected is None:  # pragma: no cover
                other_selected = []
            else:
                other_selected = selected[selected["symbol"] != symbol]["isotope"]

            collisions = 0
            for num, action in button.isotope_actions.items():
                if num in other_selected:
                    action.setIcon(QtGui.QIcon.fromTheme("folder-important"))
                    collisions += 1
                else:
                    action.setIcon(QtGui.QIcon())

            if collisions > 0:
                button.setIcon(QtGui.QIcon.fromTheme("folder-important"))
            else:
                button.setIcon(QtGui.QIcon())


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    table = PeriodicTableSelector()
    table.show()
    app.exec()
