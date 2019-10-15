 
import matplotlib.pyplot as plt

def main():
    # all_probabilities = [0.9995977, 0.9995721, 0.9995433, 0.9994998, 1.01, 0.9995902, 0.9995609, 0.98959035, 0.99162066, 0.9914842, 0.98129064, 0.99948347, 0.9855127, 0.9995136, 0.9996456, 0.9937655, 0.9877456, 0.99956614, 0.999592, 0.9995882, 0.9981185, 0.98793167, 0.9855738, 0.98129064, 0.9833717, 0.97972006, 0.97881114, 0.97994494, 0.9994936, 0.99948055, 0.9995907, 0.9996196, 0.999479, 0.99670756, 0.9995888, 0.9995672, 0.99952304, 0.9731353, 0.99933726, 0.9251323, 0.9995066, 0.9995004, 0.9995498, 0.9995332, 0.9920109, 0.99959546, 0.99539447, 0.9915531, 0.99602157, 0.9292952, 0.94930214, 0.98958194, 0.99948835, 0.999539, 0.99964416, 0.99837154, 0.9914938, 0.9994667, 0.9995529, 0.99957687, 0.99955505, 0.9995402, 0.9995573, 0.99959475, 0.9995945, 0.99955016, 0.9994554, 0.99608046, 0.99956185, 0.91671544, 0.9920346, 0.9653319, 0.9994972, 0.99954396, 0.9995396, 0.9995066, 0.9995135, 0.99905306, 0.99950075, 0.99958473, 0.9995433, 0.9951362, 0.99952054, 0.99956304, 0.9995977, 0.99953187, 0.9995876, 0.9992168, 0.9994184, 0.9930941, 0.9995946, 0.9859715, 0.9995715, 0.99952126, 0.9995834, 0.99950767, 0.9982047, 0.990226, 0.9995633, 0.9995419, 0.9995284, 0.99951196, 0.999572, 0.9995239, 0.98167783, 0.9995253, 0.9995197, 0.9995153, 0.9996451, 0.989885, 0.9995511, 0.9205841, 0.9810986, 0.99341667, 0.9995192, 0.99959964, 0.99949574, 0.9980716, 0.9924535, 0.9859698, 0.9995003, 0.99955183, 0.9995029, 0.9995865, 0.9920643, 0.97945595, 0.99958736, 0.99959093, 0.99960774, 0.99950075, 0.9672062, 0.9990741, 0.97744286, 0.9314859, 0.983788, 0.9995517, 0.99952817, 0.99958163, 0.99857414, 0.9951865, 0.9946104, 0.9995956, 0.9995789, 0.99958724, 0.9992029, 0.9995017, 0.9930288, 0.99959284, 0.9995055, 0.99950397, 0.9995522, 0.97728693, 0.9927591, 0.9881116, 0.99948585, 0.97938466, 0.9995989, 0.99952745, 0.9994822, 0.999006, 0.9995882, 0.9995079, 0.94230676, 0.58423734, 0.98762566, 0.9994974, 0.99952745, 0.9995962, 0.99950147, 0.99940217, 0.9994671, 0.99960095, 0.9995141, 0.9995485, 0.99883026, 0.9982071, 0.99959534, 0.9995708, 0.9995425, 0.9995338, 0.9995504, 0.9963677, 0.9879326, 0.9972778, 0.99357224, 0.9996202, 0.99958223, 0.999553, 0.9995931, 0.9976883, 0.9995184, 0.99960476, 0.99952793, 0.99958366, 0.99959284, 0.9864839, 0.9940434, 0.9995584, 0.9995633, 0.9995596, 0.99963295, 0.9995844, 0.9995454, 0.9912991, 0.9996307, 0.9921483, 0.9940154, 0.99951315, 0.999585, 0.9995751, 0.9925541, 0.99812204, 0.9976883, 0.9995203, 0.99956006, 0.9995389, 0.997755, 0.9960102, 0.9995363, 0.9995511, 0.99952936, 0.99958235, 0.99815184, 0.9724825, 0.99723214, 0.96565706, 0.992535, 0.9918699, 0.9995939, 0.9995234, 0.9995161, 0.99943167, 0.9950321, 0.9935808, 0.99962366, 0.9981773, 0.9995499, 0.99959534, 0.99689144, 0.99898475, 0.9995931, 0.9995995, 0.9995795, 0.9996239, 0.99942243, 0.99855536, 0.969115, 0.9903825, 0.9490735, 0.9995161, 0.999572, 0.99955064, 0.99731135, 0.99494404, 0.9892588, 0.98960567, 0.9902575, 0.94241667, 0.9994838, 0.9995285, 0.9995065, 0.99961495, 0.9994374, 0.99944264, 0.92659223, 0.99719846, 0.93103266, 0.9995328, 0.99959964, 0.999503, 0.9994087, 0.99956816, 0.99953246, 0.999617, 0.9993857, 0.9854907, 0.99925965, 0.99968505, 0.9995259, 0.99962676, 0.9995369, 0.9969656, 0.9332166, 0.9989612, 0.9995321, 0.9995351, 0.9995328, 0.9995166, 0.99468225, 0.99829465, 0.99257445, 0.9858657, 0.89230514, 0.9468957, 0.9995003, 0.9994831, 0.999521, 0.9974522, 0.99953043, 0.9994993]
    # all_probabilities = [0.99688053, 0.9970682, 0.9971065, 0.9965634, 0.99691236, 0.99684227, 0.99692863, 0.9967591, 0.9969119, 0.99623054, 0.99674857, 0.9965978, 0.99716175, 0.99650925, 0.9972365, 0.99686986, 0.9968231, 0.99738497, 0.996867, 0.9971245, 0.9974675, 0.9968399, 0.9969868, 0.9981091, 0.99283916, 0.5001166, 0.9980836, 0.9981111, 0.9983814, 0.99048656, 0.9984024, 0.9972281, 0.9965371, 0.9968919, 0.9970939, 0.9970692, 0.9978231, 0.9972844, 0.99672925, 0.9974637, 0.9964671, 0.9971023, 0.9971724, 0.99703085, 0.9976947, 0.9969618, 0.99658066, 0.99739, 0.9964226, 0.99718285, 0.996619, 0.997294, 0.9965515, 0.9964563, 0.9974968, 0.9969704, 0.9986395, 0.9980792, 0.9904197, 0.9964222, 0.99539363, 0.9968844, 0.9965373, 0.9897552, 0.9695416, 0.99700886, 0.9969214, 0.99687, 0.9970101, 0.9969168, 0.9969132, 0.9969074, 0.9969652, 0.9969861, 0.99698704, 0.9968106, 0.9969093, 0.9969214, 0.9967961, 0.9967853, 0.99687, 0.99690384, 0.9969614, 0.99696475, 0.9969452, 0.99699926, 0.99684477, 0.99687, 0.9968514, 0.9228151, 0.99646544, 0.9919314, 0.9976399, 0.9927999, 0.99018466, 0.9982621, 0.99753785, 0.99743015, 0.9970886, 0.9967614, 0.9967564]
    # all_probabilities =  [0.99688053, 0.9970682, 0.9971065, 0.9965634, 0.99691236, 0.99684227, 0.99692863, 0.9967591, 0.9969119, 0.99623054, 0.99674857, 0.9965978, 0.99716175, 0.99650925, 0.9972365, 0.99686986, 0.9968231, 0.99738497, 0.996867, 0.9971245, 0.9974675, 0.9968399, 0.9969868, 0.9981091, 0.99283916, 0.5001166, 0.9980836, 0.9981111, 0.9983814, 0.99048656, 0.9984024, 0.9972281, 0.9965371, 0.9968919, 0.9970939, 0.9970692, 0.9978231, 0.9972844, 0.99672925, 0.9974637, 0.9964671, 0.9971023, 0.9971724, 0.99703085, 0.9976947, 0.9969618, 0.99658066, 0.99739, 0.9964226, 0.99718285, 0.996619, 0.997294, 0.9965515, 0.9964563, 0.9974968, 0.9969704, 0.9986395, 0.9980792, 0.9904197, 0.9964222, 0.99539363, 0.9968844, 0.9965373, 0.9897552, 0.9695416, 0.99700886, 0.9969214, 0.99687, 0.9970101, 0.9969168, 0.9969132, 0.9969074, 0.9969652, 0.9969861, 0.99698704, 0.9968106, 0.9969093, 0.9969214, 0.9967961, 0.9967853, 0.99687, 0.99690384, 0.9969614, 0.99696475, 0.9969452, 0.99699926, 0.99684477, 0.99687, 0.9968514, 0.9228151, 0.99646544, 0.9919314, 0.9976399, 0.9927999, 0.99018466, 0.9982621, 0.99753785, 0.99743015, 0.9970886, 0.9967614, 0.9967564, 0.9967481, 0.99648523, 0.9968144, 0.99677354, 0.9967648, 0.9967512, 0.9965809, 0.99670464, 0.9969202, 0.9965282, 0.9967271, 0.99679655, 0.9968046, 0.9965078, 0.99674124, 0.9967553, 0.9967796, 0.9967614, 0.9969085, 0.9967204, 0.9967564, 0.9979534, 0.99686736, 0.98907626, 0.98735833, 0.99764234, 0.000446283, 0.99712366, 0.97920805, 0.99789315, 0.99835086, 0.99747854, 0.99749905, 0.9975788, 0.99734795, 0.9973459, 0.9972012, 0.99750465, 0.99759126, 0.9975629, 0.997459, 0.9972019, 0.9976732, 0.9976961, 0.99761367, 0.99736565, 0.9976216, 0.997595, 0.9974247, 0.9974642, 0.9974946, 0.99789447, 0.9974367, 0.99736565, 0.7573984, 0.9801349, 0.98436165, 0.94138074, 0.9979175, 0.9968182, 0.99839693, 0.9979346, 0.99483263, 0.9974111, 0.99674535, 0.99670416, 0.9966466, 0.99664766, 0.9966574, 0.99669445, 0.9964902, 0.9966016, 0.9964121, 0.9966774, 0.99665594, 0.9966487, 0.99669063, 0.9973769, 0.99664813, 0.9967236, 0.9964903, 0.99665415, 0.99665844, 0.99669194, 0.9966002, 0.9966275, 0.9964659, 0.98884124, 0.99600595, 0.9952242, 0.9952178, 0.9952454, 0.9887771, 0.99764013, 0.9980013, 0.996671, 0.9970131, 0.99732995, 0.9981451, 0.99810576,
    # 					  0.9912993311882019,
    # 					  0.9973425269126892,
    # 					  0.9972273707389832,
    # 					  0.9980694651603699,
    # 					  0.9981441497802734,
    # 					  0.9923422932624817,
    # 					  0.9966686367988586,
    # 					  0.9981151819229126,
    # 					  0.9937450289726257,
    # 					  0.9978068470954895,
    # 					  0.9981054067611694,
    # 					  0.9981698989868164,
    # 					  0.9951362013816833,
    # 					  0.9981675148010254,
    # 					  0.9973691701889038,
    # 					  0.9981278777122498,
    # 					  0.9931485056877136,
    # 					  0.9960789084434509,
    # 					  0.9942724704742432]
    all_probabilities = [0.9970265, 0.99693227, 0.9968851, 0.99647105, 0.9964774, 0.99707115, 0.99834657, 0.9968509, 0.9972908, 0.99684083, 0.99663347, 0.9962083, 0.996933, 0.99637043, 0.99728644, 0.9968136, 0.9969283, 0.9968689, 0.997166, 0.9972006, 0.9951717, 0.9969965, 0.9972958, 0.98517865, 0.99340713, 0.0017021665, 0.99845195, 0.9975299, 0.9983637, 0.9592337, 0.99721366, 0.99747026, 0.9964359, 0.99762684, 0.99764746, 0.9976221, 0.997546, 0.99726415, 0.9978218, 0.9973037, 0.998264, 0.9977239, 0.9975017, 0.9973295, 0.99780315, 0.99798214, 0.9974886, 0.99767405, 0.99758327, 0.99763966, 0.9978447, 0.99769706, 0.9977502, 0.9975904, 0.99734503, 0.9977763, 7.092282e-05, 0.99862635, 0.0001089444, 0.9459163, 0.9984615, 0.99209493, 0.9983589, 0.9980293, 0.99786806, 0.99818546, 0.9966814, 0.99689156, 0.99659216, 0.99663603, 0.9967476, 0.99678516, 0.99657273, 0.9967416, 0.9966273, 0.99674976, 0.9965276, 0.99686986, 0.9966875, 0.99655247, 0.9969253, 0.99677426, 0.9967397, 0.996591, 0.9967282, 0.9966009, 0.9967476, 0.996745, 0.9967055, 7.139643e-05, 0.998594, 0.9920962, 0.9971733, 0.9906042, 0.9945702, 0.9948597, 0.99182266, 0.99745435, 0.99749506, 0.99721485, 0.99703974, 0.99629337, 0.9965815, 0.99659616, 0.99686146, 0.99610716, 0.99673957, 0.9968303, 0.99646384, 0.99827385, 0.98662055, 0.997799, 0.9966846, 0.99563545, 0.9979938, 0.996347, 0.996225, 0.9971187, 0.9983347, 0.99631786, 0.99729425, 0.99623376, 0.9907692, 0.99548703, 0.9972511, 0.9962877, 0.9938263, 0.9981146, 0.99801755, 0.99797577, 0.99771166, 0.9972108, 0.9966744, 0.9969512, 0.9971289, 0.99689627, 0.9964876, 0.99682295, 0.9968604, 0.997276, 0.99723274, 0.9969024, 0.9967585, 0.9964593, 0.99618053, 0.99673355, 0.99616504, 0.997047, 0.9965837, 0.99646, 0.9968947, 0.9970029, 0.9966053, 0.9959912, 0.9969874, 0.99541473, 0.99678075, 0.93689483, 0.97475624, 0.9985928, 0.99816614, 0.9983724, 0.9965569, 0.99695337, 0.99716514, 0.9973484, 0.99598145, 0.99826187, 0.9950995, 0.99634874, 0.9983322, 0.9954276, 0.9968925, 0.9963475, 0.9968418, 0.9964844, 0.99833375, 0.9974064, 0.99710745, 0.9971386, 0.99773043, 0.996433, 0.9946912, 0.99737096, 0.99617285, 0.99717534, 0.9969009, 0.9970765, 0.9981021, 0.9942386, 0.8847937, 0.9924205, 0.9942204, 0.9984523, 0.99745566, 0.9974165, 0.99779177, 0.9977203, 0.99743396, 0.9976514, 0.9972767, 0.99653673, 0.9966376, 0.997703, 0.99707377, 0.9982717, 0.9965653, 0.9970456, 0.99765825, 0.9967064, 0.9973232, 0.99780864, 0.9968736, 0.9981869, 0.99753845, 0.99708647, 0.9967769, 0.9976726, 0.9974431, 0.9968959, 0.9896606, 0.98773474, 0.9985696, 0.9874857, 0.98355246, 0.99847573, 0.99783653, 0.9983393, 0.99824333, 0.99722326, 0.9982456, 0.9974638, 0.9975171, 0.9975424, 0.9975337, 0.9975815, 0.99744415, 0.99753714, 0.9975553, 0.9975171, 0.9975171, 0.9975303, 0.99751437, 0.99753976, 0.99750876, 0.9975101, 0.9975101, 0.9975101, 0.9975303, 0.9975305, 0.9975171, 0.9975132, 0.99744415, 0.9975426, 0.9985489, 0.99624544, 0.97520995, 0.99833715, 0.9980805, 0.9348263, 0.9976681, 0.9983424, 0.99613756, 0.99686277, 0.9974201, 0.99753594, 0.9973621, 0.9976267, 0.9974432, 0.99755377, 0.9974201, 0.99748933, 0.9975661, 0.9974957, 0.99753416, 0.9974728, 0.99741495, 0.9973425, 0.99755585, 0.9976343, 0.9974663, 0.99759597, 0.9974957, 0.99756324, 0.99729025, 0.99731654, 0.9973761, 0.99781275, 0.9904062, 0.9835571, 0.99602425, 0.9981583, 0.9927678, 0.9961767, 0.9134583, 0.9968516, 0.9973941, 0.9972933, 0.99709284, 0.99729925, 0.99717605]

    plt.plot(all_probabilities)
    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([0, 1])
    axes.grid()
    plt.show()

if __name__ == "__main__":
    main()