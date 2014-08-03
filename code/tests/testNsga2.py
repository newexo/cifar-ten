import NSGAII

class TestNSGA2():

    def setUp(self):
        self.experiment_data = {1: [97.0556096872, 1.36, 1.35], 2: [673.313472934, 3.25, 3.51], 3: [692.132532167, 2.11, 2.41], 4: [335.7538712506, 2.95, 3.25], 5: [2260.548634562, 3.29, 3.88], 6: [11.155794346330001, 1.69, 1.64], 7: [24.63379095394, 2.03, 2.15], 8: [409.6240564503, 1.56, 1.56], 9: [898.810324367, 3.39, 3.86], 10: [172.4582550844, 1.81, 1.84], 11: [600.1252539, 2.45, 2.44], 12: [46.217430265741996, 3.8, 3.94], 13: [651.1600640341001, 1.63, 1.44], 14: [67.2122956832, 3.48, 3.93], 15: [24.7623200337, 7.22, 7.56], 16: [79.8420417031, 2.4, 2.74], 17: [11.28336408138, 1.44, 1.52], 18: [402.092222249, 6.95, 7.45], 19: [134.94431340343002, 1.58, 1.94], 20: [111.6448272665, 2.58, 2.83], 21: [26.153356035599998, 7.41, 7.95], 22: [9.51161143383, 5.34, 5.8], 23: [85.73795069849999, 1.42, 1.49], 24: [98.0892743309, 2.72, 2.99], 25: [9.827868747707, 7.87, 7.94], 26: [26.291139551039997, 2.27, 2.32], 27: [4.919632351395, 2.24, 2.45], 28: [33.3956192493, 2.42, 2.51], 29: [30.1249961178, 6.0, 6.29], 30: [40.366883115, 7.57, 7.92], 31: [6.745760913690001, 1.46, 1.69], 32: [37.1334197839, 14.47, 15.27], 33: [70.68342299860001, 2.01, 2.12], 34: [29.7249544382, 3.23, 3.63], 35: [43.2995401502, 2.21, 2.15], 36: [23.861796232000003, 2.05, 2.1], 37: [28.03271466896, 3.96, 4.33], 38: [14.08451858363, 1.62, 1.61], 39: [19.57215631801, 7.09, 7.37], 40: [22.41226629811, 6.69, 7.32], 41: [28.5279988329, 1.34, 1.41], 42: [70.9185322364, 2.89, 3.31], 43: [32.5652491847, 1.3, 1.35], 44: [30.21138341423, 4.21, 4.53], 45: [46.4234406153, 5.55, 6.55], 46: [18.26159963612, 2.64, 2.88], 47: [24.30480483373, 5.86, 6.17], 48: [14.65161636671, 1.35, 1.23], 49: [62.63875111739, 1.35, 1.32], 50: [8.075368984539999, 1.36, 1.51], 51: [80.6095053672, 6.61, 7.32], 52: [34.100238247680004, 2.01, 2.01], 53: [21.59578280047, 3.44, 4.07], 54: [48.544654281899994, 9.3, 10.34], 55: [11.537434899800001, 1.29, 1.42], 56: [16.09890386663, 1.95, 2.22], 57: [24.5347844203, 5.69, 6.05], 58: [210.7807526505, 4.33, 4.85], 59: [7.02981651624, 1.6, 1.77], 60: [79.32228827084, 3.71, 4.09], 61: [7.632055799170001, 3.86, 4.45], 62: [40.34848696389, 1.62, 1.57], 63: [5.258120699722, 6.82, 6.97], 64: [259.732026033, 5.19, 5.9], 65: [6.23648835024, 3.04, 3.38], 66: [61.8007490873, 2.45, 2.73], 67: [47.7676234484, 2.55, 3.1], 68: [12.19665643376, 2.03, 2.03], 69: [3.786718181773, 6.74, 6.92], 70: [68.6198461453, 9.77, 10.35], 71: [92.70950179891, 1.54, 1.49], 72: [76.7373666684, 4.0, 4.33], 73: [144.0945613305, 3.45, 3.71], 74: [29.16592758494, 3.83, 4.14], 75: [44.7489674171, 1.54, 1.55], 76: [14.010271716120002, 3.76, 4.24], 77: [7.72991123597, 1.68, 1.66], 78: [54.9378365477, 1.71, 1.78], 79: [264.86362068290003, 5.94, 6.13], 80: [6.06724348465, 2.32, 2.34], 81: [64.36818371221, 1.81, 1.9], 82: [104.89610723232, 6.39, 7.03], 83: [49.2012576858, 1.44, 1.65], 84: [45.4720013658, 6.73, 7.09], 85: [199.0098458136, 3.69, 4.34], 86: [87.636128219, 4.15, 4.64], 87: [119.5288446705, 11.56, 11.92], 88: [76.9551467459, 2.47, 2.54], 89: [24.7237137516, 8.64, 9.08], 90: [14.72097647985, 3.86, 4.26], 91: [26.13976425325, 4.28, 4.95], 92: [2.421142701305, 3.44, 3.94], 93: [23.78380421801, 3.27, 3.69], 94: [164.84611951899998, 2.75, 3.02], 95: [27.42196133132, 4.34, 4.49], 96: [11.95830045147, 2.57, 2.55], 97: [3.773401848477, 1.95, 1.92], 98: [63.5184076706, 2.03, 2.01], 99: [48.1037550847, 4.41, 4.55]}

    def testFastNondominatedSort(self):
        def testRankList(expected, actual):
            self.assertEqual(3, actual[1])
            self.assertEqual(5, actual[8])
            self.assertEqual(11, actual[58])
            self.assertEqual(expected, actual)
		
        def testDominationFronts(expected, actual):
            self.assertEqual(expected, actual)
            self.assertEqual(expected[3], actual[3])
            self.assertEqual(expected[13], actual[13])
            self.assertEqual(len(expected), len(actual))

        expected = [{1: 3, 2: 11, 3: 7, 4: 10, 5: 12, 6: 2, 7: 4, 8: 5, 9: 12, 10: 5, 11: 7, 12: 7, 13: 4, 14: 8, 15: 8, 16: 7, 17: 2, 18: 14, 19: 5, 20: 8, 21: 9, 22: 5, 23: 3, 24: 8, 25: 6, 26: 5, 27: 2, 28: 6, 29: 8, 30: 9, 31: 1, 32: 10, 33: 6, 34: 6, 35: 5, 36: 4, 37: 7, 38: 3, 39: 6, 40: 7, 41: 1, 42: 8, 43: 1, 44: 8, 45: 9, 46: 5, 47: 7, 48: 1, 49: 2, 50: 1, 51: 10, 52: 4, 53: 6, 54: 10, 55: 1, 56: 4, 57: 7, 58: 11, 59: 2, 60: 9, 61: 4, 62: 3, 63: 3, 64: 12, 65: 3, 66: 7, 67: 7, 68: 3, 69: 2, 70: 11, 71: 4, 72: 9, 73: 9, 74: 7, 75: 3, 76: 4, 77: 1, 78: 4, 79: 13, 80: 2, 81: 5, 82: 11, 83: 3, 84: 9, 85: 10, 86: 10, 87: 12, 88: 7, 89: 8, 90: 5, 91: 7, 92: 1, 93: 6, 94: 9, 95: 7, 96: 3, 97: 1, 98: 5, 99: 9}, [[31, 41, 43, 48, 50, 55, 77, 92, 97], [59, 49, 6, 17, 27, 69, 80], [1, 23, 38, 62, 68, 75, 83, 63, 65, 96], [13, 71, 52, 56, 7, 36, 78, 61, 76], [8, 19, 46, 26, 35, 10, 81, 98, 22, 90], [53, 93, 28, 34, 33, 25, 39], [40, 37, 47, 57, 74, 91, 95, 66, 67, 12, 3, 11, 16, 88], [15, 89, 44, 29, 14, 42, 20, 24], [21, 45, 99, 30, 84, 60, 72, 73, 94], [32, 54, 51, 86, 4, 85], [70, 82, 2, 58], [87, 5, 9, 64], [79], [18]]]
        actual = NSGAII.fast_nondominated_sort(self.experiment_data)
        testRankList(expected[0], actual[0])
        testDominationFronts(expected[1], actual[1])

    def testCrowdingDistance(self):
        expected = {1: -0.005277881727443084, 2: -0.0389856133101291, 3: -0.11912221891050032, 4: -0.08710911662952933, 5: float("-inf"), 6: -0.0057697374019387605, 7: -0.00373786919007382, 8: -0.09215729954971238, 9: -0.7180540452514209, 10: -0.03126349023458373, 11: -0.11208814197186898, 12: -0.0064446680190158945, 13: -0.0419491286873437, 14: -0.024365739034623524, 15: -0.05838208499164267, 16: -0.015279797924385337, 17: -0.004535461063470064, 18: -0.06673137456927547, 19: -0.020323957466655242, 20: -0.021762618932308648, 21: -0.10781900390880647, 22: -0.09575401534870055, 23: -0.012742759629667278, 24: -0.021789612233681192, 25: -0.0840484815096793, 26: -0.01517860652704613, 27: -0.01018970997668605, 28: -0.01088364503008504, 29: -0.06142357996854804, 30: -0.06327375130634261, 31: -0.01577333249587775, 32: float("-inf"), 33: -0.009131566774265909, 34: -0.0291784634956522, 35: -0.016789767636670033, 36: -0.012710781551096683, 37: -0.01609771271332911, 38: -0.006787221807156954, 39: -0.03122136912850171, 40: -0.026455424368965474, 41: -0.00928121913123563, 42: -0.02711472351451418, 43: -0.009477253237482903, 44: -0.015217587074488611, 45: -0.07211367679963573, 46: -0.023556277936233383, 47: -0.03066217424610381, 48: float("-inf"), 49: -0.010066390046907504, 50: -0.00747812083927807, 51: -0.028933986398220087, 52: -0.00815847116981297, 53: -0.015735115304988338, 54: -0.17667782734839851, 55: float("-inf"), 56: -0.024298363414294602, 57: -0.04004794175604141, 58: -0.05352262541236784, 59: -0.00983764885039189, 60: -0.011575278773299533, 61: -0.01326997000875187, 62: -0.00575190129977881, 63: -0.02427620361859884, 64: -0.11231804101684652, 65: -0.03578980171485262, 66: -0.01846052412834165, 67: -0.024713138308141244, 68: -0.007318961198646914, 69: -0.03725066019629929, 70: -0.2855447103528305, 71: -0.011665672469770953, 72: -0.017801315497686376, 73: -0.028385022591023552, 74: -0.015766178288676946, 75: -0.005328516667542428, 76: -0.01621156654139609, 77: -0.007597673833036753, 78: -0.01967007823498536, 79: -0.05283504157859012, 80: -0.016706950867734376, 81: -0.01795597954749839, 82: -0.06083226039757647, 83: -0.005773139579648826, 84: -0.025099198487531782, 85: -0.04296860987285665, 86: -0.040388068636175355, 87: -0.7173463169955279, 88: -0.011580975333273858, 89: -0.27878256248551997, 90: -0.014638433556744011, 91: -0.07738453481020385, 92: float("-inf"), 93: -0.00937482384614046, 94: -0.03329380270680644, 95: -0.012539055900694544, 96: -0.016100872380293937, 97: -0.008006092887506535, 98: -0.003707822325512265, 99: -0.07267051591403287}
        actual = NSGAII.crowding_distance(self.experiment_data)
        self.assertEqual(expected, actual)
        
