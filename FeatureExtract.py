import math
from collections import Counter
from Bio import SeqIO
import numpy as np

def minSequenceLength(fastas):
	minLen = 10000000000
	for i in fastas:
		if minLen > len(i[1]):
			minLen = len(i[1])
	return minLen

# AAC 20D
def AAC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']

    for aa in AA:
        header.append('AAC_' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = [name]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings

# DPC 400D
def DPC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for aa in diPeptides:
        header.append('DPC_' + aa)
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

# CKSAAP 400D
def CKSAAP(fastas, gap=1):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']

    for aa in diPeptides:
        header.append('CKSAAP_' + aa + '.gap' + str(gap))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        myDict = {}
        for pair in diPeptides:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(sequence)):
            index2 = index1 + gap + 1
            if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                myDict[sequence[index1] + sequence[index2]] += 1
                sum += 1
        for pair in diPeptides:
            code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

def ASDC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']

    for aa in diPeptides:
        header.append('ASDC_' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        myDict = {}
        for pair in diPeptides:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(sequence)):
            for index2 in range(index1 + 1, len(sequence)):
                if sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] += 1
                    sum += 1
        for pair in diPeptides:
            code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

# GAAC 5D
def GAAC(fastas):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }
    groupKey = group.keys()

    encodings = []
    header = ['#']
    for key in groupKey:
        header.append('GAAC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key] / len(sequence))
        encodings.append(code)

    return encodings

# GDPC 25D
def GDPC(fastas):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    header = ['#']
    for key in dipeptide:
        header.append('GDPC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]

        code = [name]
        myDict = {}
        for t in dipeptide:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 1):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in dipeptide:
                code.append(0)
        else:
            for t in dipeptide:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return encodings

# GTPC 125D
def GTPC(fastas):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()

    triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    header = ['#']
    for key in triple:
        header.append('GTPC_' + key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]

        code = [name]
        myDict = {}
        for t in triple:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 2):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] = myDict[index[
                                                                                                                  sequence[
                                                                                                                      j]] + '.' +
                                                                                                              index[
                                                                                                                  sequence[
                                                                                                                      j + 1]] + '.' +
                                                                                                              index[
                                                                                                                  sequence[
                                                                                                                      j + 2]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in triple:
                code.append(0)
        else:
            for t in triple:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return encodings

# CKSAAGP 25D
def CKSAAGP(fastas, gap=1):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0
    if minSequenceLength(fastas) < gap + 2:
        print('Error: all the sequence length should be greater than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    AA = 'ARNDCQEGHILKMFPSTWYV'
    groupKey = group.keys()
    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key
    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)
    encodings = []
    header = ['#']

    for p in gPairIndex:
        header.append('CKSAAGP_' + p + '.gap' + str(gap))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + gap + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                    sequence[p2]]] + 1
                sum = sum + 1
        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)
        encodings.append(code)
    return encodings

# CTDC 39D
def CTDC(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append('CTDC_' + p + '.G' + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for p in property:
            sum1 = 0
            for aa in group1[p]:
                sum1 = sum1 + sequence.count(aa)
            c1 = sum1 / len(sequence)

            sum2 = 0
            for aa in group2[p]:
                sum2 = sum2 + sequence.count(aa)
            c2 = sum2 / len(sequence)

            sum3 = 0
            for aa in group3[p]:
                sum3 = sum3 + sequence.count(aa)
            c3 = sum3 / len(sequence)

            code = code + [c1, c2, c3]
        encodings.append(code)
    return encodings

# CTDT 39D
def CTDT(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append('CTDT_' + p + '.' + tr)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
        encodings.append(code)
    return encodings

# CTDD 195D
def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence))
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append('CTDD_' + p + '.' + g + '.residue' + d)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for p in property:
            code = code + Count(group1[p], sequence) + Count(group2[p], sequence) + Count(group3[p], sequence)
        encodings.append(code)
    return encodings

# Fasta
def ReadFileFromFasta(filepath):
    seq = []
    for seq_record in SeqIO.parse(filepath, "fasta"):
        seq.append(['>' + seq_record.id.strip(), str(seq_record.seq).strip()])
    return seq

def FeatureGenerator(fastas):
    labels = []

    FeatureDict = {}
    FeatureNameDict = {}

    for i in fastas:
        name = i[0]
        if str(name).startswith('>') and str(name).find('_URS') != -1:
            labels.append(0)
        else:
            labels.append(1)

    aac = np.array(AAC(fastas))
    dpc = np.array(DPC(fastas))
    asdc = np.array(ASDC(fastas))
    c1saap = np.array(CKSAAP(fastas, 1))
    c2saap = np.array(CKSAAP(fastas, 2))
    c3saap = np.array(CKSAAP(fastas, 3))
    gaac = np.array(GAAC(fastas))
    gdpc = np.array(GDPC(fastas))
    gtpc = np.array(GTPC(fastas))
    c1saagp = np.array(CKSAAGP(fastas, 1))
    c2saagp = np.array(CKSAAGP(fastas, 2))
    c3saagp = np.array(CKSAAGP(fastas, 3))
    ctdc = np.array(CTDC(fastas))
    ctdt = np.array(CTDT(fastas))
    ctdd = np.array(CTDD(fastas))

    FeatureDict['aac'] = np.array(aac[1:, 1:], dtype=float)
    FeatureNameDict['aac'] = aac[:1, :][0]

    FeatureDict['dpc'] = np.array(dpc[1:, 1:], dtype=float)
    FeatureNameDict['dpc'] = dpc[:1, :][0]

    FeatureDict['asdc'] = np.array(asdc[1:, 1:], dtype=float)
    FeatureNameDict['asdc'] = asdc[:1, :][0]

    FeatureDict['c1saap'] = np.array(c1saap[1:, 1:], dtype=float)
    FeatureNameDict['c1saap'] = c1saap[:1, :][0]

    FeatureDict['c2saap'] = np.array(c2saap[1:, 1:], dtype=float)
    FeatureNameDict['c2saap'] = c2saap[:1, :][0]

    FeatureDict['c3saap'] = np.array(c3saap[1:, 1:], dtype=float)
    FeatureNameDict['c3saap'] = c3saap[:1, :][0]

    FeatureDict['gaac'] = np.array(gaac[1:, 1:], dtype=float)
    FeatureNameDict['gaac'] = gaac[:1, :][0]

    FeatureDict['gdpc'] = np.array(gdpc[1:, 1:], dtype=float)
    FeatureNameDict['gdpc'] = gdpc[:1, :][0]

    FeatureDict['gtpc'] = np.array(gtpc[1:, 1:], dtype=float)
    FeatureNameDict['gtpc'] = gtpc[:1, :][0]

    FeatureDict['c1saagp'] = np.array(c1saagp[1:, 1:], dtype=float)
    FeatureNameDict['c1saagp'] = c1saagp[:1, :][0]

    FeatureDict['c2saagp'] = np.array(c2saagp[1:, 1:], dtype=float)
    FeatureNameDict['c2saagp'] = c2saagp[:1, :][0]

    FeatureDict['c3saagp'] = np.array(c3saagp[1:, 1:], dtype=float)
    FeatureNameDict['c3saagp'] = c3saagp[:1, :][0]

    FeatureDict['ctdc'] = np.array(ctdc[1:, 1:], dtype=float)
    FeatureNameDict['ctdc'] = ctdc[:1, :][0]

    FeatureDict['ctdt'] = np.array(ctdt[1:, 1:], dtype=float)
    FeatureNameDict['ctdt'] = ctdt[:1, :][0]

    FeatureDict['ctdd'] = np.array(ctdd[1:, 1:], dtype=float)
    FeatureNameDict['ctdd'] = ctdd[:1, :][0]

    return FeatureDict, np.array(labels).astype(int), FeatureNameDict
