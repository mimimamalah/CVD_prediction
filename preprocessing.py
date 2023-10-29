import numpy as np
import helpers

collp2 = ["HLTHPLN1", "MEDCOST", "BPMEDS", "BLOODCHO", "TOLDHI2","CVDSTRK3", 
                   "ASTHMA3", "ASTHNOW","CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3",
                   "ADDEPEV2","CHCKIDNY", "CPDEMO1", "VETERAN3", "INTERNET", "PREGNANT",
                   "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES",
                   "DIFFALON", "SMOKE100","STOPSMK2", "EXERANY2","LMTJOIN3", "ARTHDIS2", 
                   "FLUSHOT6", "PNEUVAC3","HIVTST6", "PDIABTST", "INSULIN", "DIABEYE", 
                   "DIABEDU","VIGLUMA2","VIMACDG2","CIMEMLOS","CDDISCUS","WTCHSALT",
                   "DRADVISE","ASATTACK","HADMAM","HADPAP2","HPVTEST","HADHYST2","PROFEXAM",
                   "BLDSTOOL","HADSIGM3","HADSGCO1","PCPSAAD2","PCPSADI1","PCPSARE1","PSATEST1",
                   "_PA30021","_PASTRNG","_PASTAE1"]


def dataPreprocess(x_train, x_test, y_train, columns, feature_names):
    ##defines if Yes-No question are one Hot encoded or No->0 , Yes->1 encoded
    oneHotp2 = True

    
    collumns_to_delete = data_cleaning_NaN(x_train, columns, threshold=0.6)
    collumns_to_delete += get_collumns_to_delete(collumns_to_delete,oneHotp2)

    x_new_train = np.copy(x_train)
    x_append_train = np.empty((x_train.shape[0], 0))

    x_new_test = np.copy(x_test)
    x_append_test = np.empty((x_test.shape[0], 0))

    x_append_train ,x_append_test=  oneHotEncoding(feature_names, x_append_train, x_append_test, x_train, x_test)
    x_append_train ,x_append_test=  process2bis(feature_names, x_append_train, x_append_test, x_train, x_test, oneHotp2)

    variables_for_process_HLTH = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process_HLTH, process_HLTH)

    variables_for_process3 = ["CHECKUP1", "CHOLCHK"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process3, process_3)

    variables_for_process4 = ["DIABAGE2"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process4, process_4)

    variables_for_process5 = ["CHILDREN"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process5, process_5)

    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,["WEIGHT2"], process_6_weight)
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,["HEIGHT3"], process_6_height)

    variables_for_process7 = ["ALCDAY5", "EXEROFT1", "EXEROFT2", "STRENGTH"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process7, process_7)

    variables_for_process8 = ["AVEDRNK2", "DRNK3GE5", "MAXDRNKS"]
    indice_ALCDAY5 = [
        i for i, item in enumerate(feature_names) if item.find("ALCDAY5") != -1
    ][0]
    for col in variables_for_process8:
        indice = [i for i, item in enumerate(feature_names) if item.find(col) != -1][0]
        x_new_train[:, indice] = process_8(
            x_train[:, indice], x_train[:, indice_ALCDAY5]
        )
        x_new_test[:, indice] = process_8(x_test[:, indice], x_test[:, indice_ALCDAY5])

    variables_for_process9 = ["FRUITJU1","FRUIT1","FVBEANS","FVGREEN","FVORANG","VEGETAB1"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process9, process_9)

    variables_for_process10 = ["EXERHMM1", "EXERHMM2"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process10, process_10)

    variables_for_process11 = ["BLDSUGAR", "FEETCHK2","LONGWTCH"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process11, process_11)

    variables_for_process12 = ["DOCTDIAB", "CHKHEMO3", "FEETCHK"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process12, process_12)

    variables_for_process13 = ["USENOW3","ARTHSOCL","SMOKDAY2","_PA150R2","_PA300R2","_LMTACT1","_LMTWRK1"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process13, process_13)

    variables_for_process14 = ["JOINPAIN"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process14, process_14)

    variables_for_process15 = ["EYEEXAM","ARTTODAY","SCNTMNY1","SCNTMEL1","SCNTPAID","_EDUCAG","_INCOMG","_SMOKER3","_PACAT1","CHOLCHK"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process15, process_15)


    variables_for_process16 = ["SEATBELT","CDHOUSE","CDASSIST","CDHELP","CDSOCIAL","HOWLONG",
                           "LASTPAP2","HPLSTTST","LENGEXAM","LSTBLDS3","LASTSIG3","PSATIME","PCPSARS1",
                           "EMTSUPRT","LSATISFY"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process16, process_16)

    variables_for_process17 = ["LASTSMK2"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process17, process_17)

    variables_for_process18 = ["ASTHMAGE"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process18, process_18)

    variables_for_process19 = ["SCNTWRK1", "SCNTLWK1"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process19, process_19)

    variables_for_process20 = ["ADPLEASR","ADDOWN","ADSLEEP","ADENERGY","ADEAT1","ADFAIL","ADTHINK","ADMOVE"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process20, process_20)

    variables_for_process21 = ["GRENDAY_","ORNGDAY_","VEGEDA1_","FTJUDA1_","FRUTDA1_","BEANDAY_","_FRUTSUM","_VEGESUM","PADUR1_","PADUR2_",
                           "_MINAC11","_MINAC21","PAMIN11_","PAMIN21_","PA1MIN_","PAVIG11_","PAVIG21_","PA1VIGM_"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process21, process_21)

    variables_for_process22 = ["METVL11_", "METVL21_"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process22, process_22)

    variables_for_process23 = ["MAXVO2_", "FC60_", "PAFREQ1_", "PAFREQ2_", "STRFREQ_"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process23, process_23)

    variables_for_process24 = ["_BMI5"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process24, process_24)

    variables_for_process25 = ["BPHIGH4", "DIABETE3", "PREDIAB1"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process25,process_25)

    variables_for_process26 = ["PERSDOC2"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process26,process_26)


    variables_for_process_cont = ["_STRWT","_RAWRAKE","_WT2RAKE"]
    proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,variables_for_process_cont, process_cont)


    

    indice_to_delete = [feature_names.index(item) for item in collumns_to_delete]

    indice_to_delete.sort()


    x_new_del_train = np.delete(x_new_train, indice_to_delete, 1)
    x_new_del_test = np.delete(x_new_test, indice_to_delete, 1)


    ## Normalize the data (We do the normalization here because oneHot encoded collumns don't need to be normalized )

    for i in range(x_new_del_train.shape[1]):
        mean_train =    x_new_del_train[:,i].mean()
        std_train =     x_new_del_train[:,i].std()
        mean_test =     x_new_del_test[:,i].mean()
        std_test =      x_new_del_test[:,i].std()
        if(mean_train>1.1):
            x_new_del_train[:,i] = (x_new_del_train[:,i]-mean_train)/std_train
            x_new_del_test[:,i] = (x_new_del_test[:,i]-mean_test)/std_test


    degree = 3 

    N_train = x_new_del_train.shape[0]
    D_train = x_new_del_train.shape[1]

    N_test = x_new_del_test.shape[0]
    D_test = x_new_del_test.shape[1]

    x_new_del_train = build_poly(x_new_del_train,degree)
    x_new_del_test = build_poly(x_new_del_test,degree)

    x_new_del_train = x_new_del_train.reshape(N_train,degree*D_train)
    x_new_del_test = x_new_del_test.reshape(N_test,degree*D_test)


    ## Append all the oneHot encoded COllumns

    x_train_preprocess = np.hstack((x_new_del_train, x_append_train))
    x_test_preprocess = np.hstack((x_new_del_test, x_append_test))


    ## Remove Collumns with 0 variance 
    x_train_preprocess, to_delete = remove_zero_variance_columns(x_train_preprocess)
    x_test_preprocess = np.delete(x_test_preprocess, to_delete, axis=1 )


    
    ## Remove Correalted COllumns 
    x_train_preprocess, to_remove = remove_correlated_columns(x_train_preprocess)
    x_test_preprocess = np.delete(x_test_preprocess, to_remove, axis=1)



    # Adds a collumns of ones to encode a bias terms 
    ones_column_train = np.ones((x_train_preprocess.shape[0], 1))
    x_train_preprocess = np.hstack((ones_column_train, x_train_preprocess))

    ones_column_test = np.ones((x_test_preprocess.shape[0], 1))
    x_test_preprocess = np.hstack((ones_column_test, x_test_preprocess))
    
    #Calculate the correlations between x_train_preprocess
    correlations = np.corrcoef(x_train_preprocess, y_train, rowvar=False)[:-1, -1]

    #Determine the number of features to remove (10% of total features)
    num_features_to_remove = int(0.1 * x_train_preprocess.shape[1])
    
    #Identify the indices of the least correlated features to remove
    indices_to_remove = np.argpartition(np.abs(correlations), num_features_to_remove)[:num_features_to_remove]

    # Step 4: Remove the least correlated features
    x_train_preprocess = np.delete(x_train_preprocess, indices_to_remove, axis=1)
    x_test_preprocess = np.delete(x_test_preprocess, indices_to_remove, axis=1)

    assert np.count_nonzero(np.isnan(x_train_preprocess)) == 0
    assert np.count_nonzero(np.isnan(x_test_preprocess)) == 0
    return x_train_preprocess, x_test_preprocess, y_train


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    rep =[]
    for xi in x:
        temp =[]
        for d in range(degree):
            temp.append(pow(xi,d+1)) # We don't encode d=0 since we already add the 1's collumns 
        rep.append(temp)
    return np.array(rep)

def proccess_columns(x_new_train,x_new_test,x_train,x_test,feature_names,list_of_collumns_to_process,processing_function):
    """
    Apply the processing function to the list of collumns

    Then standardize the input and store it in a new array

    Args :
        x_train (Numpy (N,D)) arary : input train data to preprocess
        x_test (numpy (_,D)) array : input train data to preprocess
        list_of_collumns_to_process : list of collumns to apply the processing function 
        processing_function : function to be applied on each collumn

        x_new_train (Numpy (N,D)) : train data after the processing
        x_new_test (Numpy (_,D)) : test data after the processing

        feature_names : List of all collumns in the datasets


    """
    for col in list_of_collumns_to_process:
        tab = [i for i, item in enumerate(feature_names) if item.find(col) != -1]
        if len(tab) > 0:
            indice = tab[0]
            x_new_train[:, indice] = standardize(processing_function(x_train[:, indice]))
            x_new_test[:, indice] = standardize(processing_function(x_test[:, indice]))


def one_hot_encoding(collumn):
    """
    Args :
        collumn (strings) : collumn to one hot encode
    Returns
        A (N,max(coll)+1) array: One hot encoded version of the collumn

    """

    collumn[np.isnan(collumn)] = 0
    int_coll = collumn.astype(int)
    num_classes = max(int_coll) + 1
    result = np.eye(num_classes)[int_coll]

    return result


def one_hot_encoding_special(collumn, num_max, skip):
    """
    Args :
        collumn (strings) : collumn to one hot encode
        num_max (Int) : the maximum number that should be oneHot encoded
        skip (List) : list of numbers that should be mapped to 0
    Returns
        A (N,num_max+1) array: One hot encoded version of the collumn

    """

    collumn[np.isnan(collumn)] = 0

    collumn_as_int = collumn.astype(int)
    num_classes = num_max + 1
    collumn_as_int[np.isin(collumn_as_int, skip)] = 0
    result = np.eye(num_classes)[collumn_as_int]

    return result


def data_cleaning_NaN(x_train, columns, threshold=0.75):
    """
        Return a list of collumn with Too many Nan values


    Args :
        x_train (N,D) numpy array : The training dataset
        threshold [0,1] :Percentage of Nan values Beyond which the column will be deleted
    Returns
        collumns_to_delete (list String) : List of collumns to delete

    """

    collumns_to_delete = []
    N = x_train.shape[0]
    D = x_train.shape[1]

    for i in range(D - 1):
        if len(np.where(np.isnan(x_train[:, i]))[0]) > N * threshold:
            collumns_to_delete.append(columns[i])

    return collumns_to_delete

def remove_zero_variance_columns(x) :
    
    variances = np.var(x, axis=0)
    zero_variance_columns = np.where(variances == 0)[0]
    x = np.delete(x, zero_variance_columns, axis=1)
    
    return x, zero_variance_columns


def remove_correlated_columns(x , threshold: float = 0.9) :
    """
    Removes correlated columns, leaving only one column for each originally correlated sets of columns.

    Arguments
    ---------
    x: np.ndarray
        The matrix to be cleaned.

    threshold: float
        The maximum percentage of correlation allowed (default = 0.9).

    Returns
    -------
    clean_x: np.ndarray
        The cleaned matrix.

    to_keep: nd.array
        Boolean array of the kept columns.
    """

    assert 0 <= threshold <= 1
    _, to_remove = np.where(np.triu(np.corrcoef(x.T), 1) > threshold)
    to_remove = list(set(to_remove))
    return np.delete(x, to_remove, axis=1), to_remove


def get_collumns_to_delete(collumns_to_delete,oneHotp2):



    array_to_drop_Useless = ["_STATE","_PSU","SEQNO","CTELENUM","STATERES","CELLFON3","DISPCODE","PVTRESD1","CTELNUM1","CELLFON2",
                             "PVTRESD2","LANDLINE","HHADULT","NUMHHOL2","NUMADULT" , "NUMMEN" , 
                             "NUMWOMEN","RCSGENDR","RCSRLTN2","QSTVER","QSTLANG","MSCODE","_STSTR",                             
                             "_CHISPNC","_CLLCPWT","_DUALCOR","HTIN4","WTKG3","HTM4",
                             "_MISFRTN","_MISVEGN","_FRTRESP","_VEGRESP","_FRT16","_VEG23","_FRUITEX","_VEGETEX",
                             "PAMISS1_","_RFSEAT2","_AIDTST3",
                             "FLSHTMY2","FEETCHK2","FEETCHK","CAREGIV1","EXRACT11","EXRACT21"]

    array_to_drop_redundant = ["IDATE","FMONTH","IYEAR"] #IYEAR is useless , 99% in 2015

    array_to_drop_too_many_missing =["LADULT","COLGHOUS","CADULT","CCLGHOUS","CSTATE","NUMPHON2", "CRGVREL1","CRGVLNG1"
                                     ,"CRGVHRS1","CRGVPRB1","CRGVPERS", "CRGVHOUS","CRGVMST2","CRGVEXPT","HIVTSTD3","VIDFCLT2",
                                     "VIREDIF3","VIPRFVS2","VINOCRE2","VIEYEXM2","VIINSUR2","VICTRCT4","ASERVIST","ASDRVIST",
                                     "ASRCHKUP","ASACTLIM","ASYMPTOM","ASNOSLEP","ASTHMED3","ASINHALR","PCPSADE1","PCDMDECN"]

    # Collumns that are oneHot encoded should be removed from the dataset
    collumns_to_delete_from_one_hot =["IMONTH","IDAY",
                                  "SEX","MARITAL","EDUCA","RENTHOM1","EMPLOY1","INCOME2","GENHLTH",
                                  "PERSDOC2","SMOKDAY2",
                                  "BPHIGH4","DIABETE3","PREDIAB1",
                                  "HAREHAB1","STREHAB1","CVDASPRN","ASPUNSAF","RLIVPAIN","RDUCHART","RDUCSTRK","ARTHWGT","ARTHEXER","ARTHEDU",
                                  "TETANUS","HPVADVC2","HPVADSHT","SHINGLE2","SCNTLPAD","SXORIENT","TRNSGNDR","CASTHDX2","CASTHNO2","MISTMNT",
                                  "ADANXEV","_CRACE1","_CPRACE","_RFHLTH","_HCVU651","_RFHYPE5","_CHOLCHK","_RFCHOL","_LTASTH1","_CASTHM1",
                                  "_ASTHMS1","_DRDXAR1","_PRACE1","_MRACE1","_HISPANC","_RACE","_RACEG21","_RACEGR3","_RACE_G1",
                                  "_AGEG5YR","_AGE65YR","_BMI5CAT","_RFBMI5","_CHLDCNT","_RFSMOK3","DRNKANY5","_RFBING5","_RFDRHV5","_FRTLT1","_VEGLT1","_TOTINDA","ACTIN11_","ACTIN21_",
                                  "_PAINDX1","_PAREC1","IMFVPLAC","WHRTST10","_DUALUSE","_AGE_G","_LMTSCL1","_RFSEAT3","_FLSHOT6","_PNEUMO2"
                                  ]

    collumns_to_delete = collumns_to_delete+array_to_drop_Useless+ array_to_drop_redundant+ array_to_drop_too_many_missing+ collumns_to_delete_from_one_hot
    if oneHotp2:
        collumns_to_delete += collp2

    return collumns_to_delete


def oneHotEncoding(feature_names, x_append_train, x_append_test, x_train, x_test):
    collumn_to_oneHotEncode = ["IMONTH","IDAY","SEX","_DRDXAR1","_RACE_G1","_BMI5CAT","ACTIN11_","ACTIN21_"]

    collumn_to_oneHotencode_special = [("MARITAL",6,[9]),("EDUCA",6,[9]),("RENTHOM1",3,[7,9]),("EMPLOY1",8,[9]),("INCOME2",8,[77,99]),
                                       ("GENHLTH",5,[7,9]),("HAREHAB1",2,[7,9]),("STREHAB1",2,[7,9]),("CVDASPRN",2,[7,9]),("ASPUNSAF",3,[7,9]),
                                       ("RLIVPAIN",2,[7,9]),("RDUCHART",2,[7,9]),("RDUCSTRK",2,[7,9]),("ARTHWGT",2,[7,9]),("ARTHEXER",2,[7,9]),
                                       ("ARTHEDU",2,[7,9]),("TETANUS",4,[7,9]),("HPVADVC2",3,[7,9]),("HPVADSHT",3,[77,99]),("SHINGLE2",2,[7,9]),
                                       ("SCNTLPAD",4,[7,9]),("SXORIENT",4,[7,9]),("TRNSGNDR",4,[7,9]),("CASTHDX2",2,[7,9]),("CASTHNO2",2,[7,9]),
                                       ("MISTMNT",2,[7,9]),("ADANXEV",2,[7,9]),("_CRACE1",7,[77,99]),("_CPRACE",7,[77,99]),("_RFHLTH",2,[9]),
                                       ("_HCVU651",2,[9]),("_RFHYPE5",2,[9]),("_CHOLCHK",3,9),("_RFCHOL",2,[9]),("_LTASTH1",2,9),("_CASTHM1",2,[9]),
                                       ("_ASTHMS1",3,[9]),("_PRACE1",8,[77,99]),("_MRACE1",7,[77,99]),("_HISPANC",2,[9]),("_RACE",8,[9]),("_RACEG21",2,[9]),
                                       ("_RACEGR3",5,[9]),("_AGEG5YR",13,[14]),("_AGE65YR",2,[3]),("_RFBMI5",2,[9]),("_CHLDCNT",6,[9]),("_RFSMOK3",2,[9]),
                                       ("DRNKANY5",2,[7,9]),("_RFBING5",2,[9]),("_RFDRHV5",2,[9]),("_FRTLT1",2,[9]),("_VEGLT1",2,[9]),("_TOTINDA",2,[9]),
                                       ("_PAINDX1",2,[9]),("_PAREC1",4,[9]),("PERSDOC2",3,[7,9]),("SMOKDAY2",3,[7,9]),("BPHIGH4",4,[7,9]),("DIABETE3",4,[7,9]),
                                       ("PREDIAB1",3,[7,9]),("IMFVPLAC",11,[77,99]),("WHRTST10",9,[77,99]),("_DUALUSE",2,[9]),("_AGE_G",6,[]),("_LMTSCL1",4,[9]),
                                       ("_RFSEAT3",2,[9]),("_FLSHOT6",2,[9]),("_PNEUMO2",2,[9])]

    for col in collumn_to_oneHotEncode:        
        indice = [i for i, item in enumerate(feature_names) if item.find(col) != -1][0]

        encoded_train = one_hot_encoding(x_train[:, indice])
        x_append_train = np.hstack((x_append_train, encoded_train))

        encoded_test = one_hot_encoding(x_test[:, indice])
        x_append_test = np.hstack((x_append_test, encoded_test))

    for col, num_max, skip in collumn_to_oneHotencode_special:
        indice = [i for i, item in enumerate(feature_names) if item.find(col) != -1][0]
        encoded_train = one_hot_encoding_special(x_train[:, indice], num_max, skip)
        x_append_train = np.hstack((x_append_train, encoded_train))

        encoded_test = one_hot_encoding_special(x_test[:, indice], num_max, skip)
        x_append_test = np.hstack((x_append_test, encoded_test))
    return x_append_train ,x_append_test


########################################################################################################################################################################################################################
########################################################################################################################################################################################################################

## PROCESS , EACH PROCESS TAKES CARE OF A TYPE OF DATA  :


def process_2(column):
    """
    Process for values that have as answer Yes or No
    We're going to replace No by 0 and yes by 1, since yes is already by 1
    For values 7 which corresponds to don't know or not sure, 9 for refused, and BLANK for missing values, we're going to take the median

    """
    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 2:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 1]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process2bis(feature_names, x_append_train, x_append_test, x_train, x_test,oneHot=True):
    """
        Process data with
        1 = Yes
        2 = No
        7 = Don't Know/Note sure
        9 = refused

    Args:
       oneHot : Boolean decide if the data will be oneHot encode or lineaerly encoded

    """

    if oneHot:
        process_2_oneHot_special = [(item, 2, [7, 9]) for item in collp2]
        for col, num_max, skip in process_2_oneHot_special:
            indice = [
                i for i, item in enumerate(feature_names) if item.find(col) != -1
            ][0]
            encoded_train = one_hot_encoding_special(x_train[:, indice], num_max, skip)
            x_append_train = np.hstack((x_append_train, encoded_train))

            encoded_test = one_hot_encoding_special(x_test[:, indice], num_max, skip)
            x_append_test = np.hstack((x_append_test, encoded_test))
        return x_append_train ,x_append_test
    else:
        proccess_columns(collp2, process_2)
        return np.empty((x_train.shape[0], 0)),np.empty((x_train.shape[0], 0))
    


def process_HLTH(column):
    """
    Process for PHYSHLTH, MENTHLTH, POORHLTH
    Possible Values :
        1-30 : Number of days
        88 None
        77 Don't Know
        99 refused
    Treatment
        The values corresponds to days, between 1-30 it is already good.
        We're going to replace all 88 values by 0 because it corresponds to None which is 0 days,
        and we will assume that those who refused or did not answer have median values.

    """
    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 88:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 30]
    median = np.nanmedian(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_3(column):
    """
    Possible Values :
        1-4 Increasing time period
        7 = Don't Know/Note sure
        8 = NEVER
        9 = refused
    Treatment :
        Mapping never to 5 and 7,9 and NaN to the median value

    """
    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 8:
            new_column[i] = 5

    filtered_elements = [x for x in new_column if 0 <= x <= 5]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_4(column):
    """
    Possible Values :
        1-97 age
        98 Don't know/not sure
        99 refused
    Treatment :
        98n99 and NaN are mapped to the median age
    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if 1 <= x <= 97]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 98) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_5(column):
    """
    Process the children data

    Possible Values :
        1-87 : Number of children
        88 : None
        99 : Refused

    Treatment :
        88 mapped to 0
        98 and NaN to the median

    """
    new_column = column
    for i in range(len(new_column)):
        if new_column[i] == 88:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 87]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 98) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_6_weight(column):
    """
    Convert the weight from pounds to kilograms
    """
    new_column = column.copy()

    for i in range(len(new_column)):
        if new_column[i] >= 50 and new_column[i] <= 999:
            new_column[i] = new_column[i] * 0.453592
        elif new_column[i] >= 9000 and new_column[i] <= 9998:
            new_column[i] = new_column[i] - 9000

    filtered_elements = [x for x in new_column if 0 <= x <= 999]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 7777)
            or (new_column[i] == 9999)
            or np.isnan(new_column[i])
        ):
            new_column[i] = median

    return new_column


def process_6_height(column):
    """
    Convert the Height from inches to cm
    """
    new_column = column.copy()

    for i in range(len(new_column)):
        if new_column[i] >= 200 and new_column[i] <= 711:
            feet = new_column[i] // 100
            inch = (new_column[i] - feet * 100) + feet * 12
            new_column[i] = inch * 2.54
        elif new_column[i] >= 9000 and new_column[i] <= 9998:
            new_column[i] = new_column[i] - 9000

    filtered_elements = [x for x in new_column if 0 <= x <= 999]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 7777)
            or (new_column[i] == 9999)
            or np.isnan(new_column[i])
        ):
            new_column[i] = median

    return new_column


def process_7(column):
    """
        Process values "time per week/month


    Possible Values :
        101-199 : days per week
        201-299 : Days in past 30 days
        777 : Don't know/Not sure
        888 Nothing during the last 30 days
        999 Refused
    Treatement
        Map everything to "per month" (week*4)
        Nothing (888) will be mapped to 0
        777,999 and NaN will be mapped to the mean
    """

    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] >= 101 and new_column[i] <= 199:
            new_column[i] = (new_column[i] - 100) * 4
        elif new_column[i] >= 201 and new_column[i] <= 299:
            new_column[i] = new_column[i] - 200
        elif new_column[i] == 888:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 99]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 777) or (new_column[i] == 999) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_8(column, column_ALCDAY5):
    """
    Preprocess data related to Alcohol Consumption

    Prossible Values :
        1- 76 : Number of Drinks
        77 : Don't Know/Not Sure
        99 REfused
    Treatment
        Here , question were asked only iif you have had at least one drink this month,so blank values will be mapped to 0
        Don't know, refused and NaN are mapped to the media


    """
    new_column = column.copy()
    for i in range(len(new_column)):
        if np.isnan(new_column[i]) and column_ALCDAY5[i] == 888:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 76]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_9(column):
    """

    Preprocess data for Fruit / vegetable consumption

    Possibles values :
        101-199     : Times per day
        201 -299    : Times per week
        300         : Less than one time per month
        301 - 399   : Times per month
        555         : Never
        777         : Don't know
        999         : Refused
    Treatment
        Mapping everything to time per month (week*4) and (day$30)
        mapping less than one time and NEVER to 0
        mapping refused/ Don't know to the median


    """

    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] >= 101 and new_column[i] <= 199:
            new_column[i] = (new_column[i] - 100) * 30
        elif new_column[i] >= 201 and new_column[i] <= 299:
            new_column[i] = (new_column[i] - 200) * 4
        elif (new_column[i] == 300) or (new_column[i] == 555):
            new_column[i] = 0
        elif new_column[i] >= 301 and new_column[i] <= 399:
            new_column[i] = new_column[i] - 300

    filtered_elements = [x for x in new_column if 0 <= x <= 99]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 777) or (new_column[i] == 999) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_10(column):
    """
    Preprocess Physical activity data

    Possibles Values :
        1-759   : Hours and Minutes
        777     : Don't Know/Not Sure
        800-959 : Hours and minutes
        999     : Refused

    Treatment
        Convert Everything into minutes
        Don't KNow / Refused are mapped to NaN

    """

    new_column = column.copy()
    for i in range(len(new_column)):
        if (new_column[i] >= 1 and new_column[i] <= 759) or (
            new_column[i] >= 800 and new_column[i] <= 959
        ):
            first_digit = new_column[i] // 100
            digits2_3 = new_column[i] % 100
            new_column[i] = first_digit * 60 + digits2_3

    filtered_elements = [x for x in new_column if 0 <= x <= 600]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 777) or (new_column[i] == 999) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_11(column):
    """
    Preprocess Diabetes data


    Possibles Values :
        101-199     : Times per day
        201 - 299   : Times per week
        301 - 399   : Times per month
        401 - 499   : Times per year
        555         : all
        777         : Don't know / Not sure
        888         :Never
        999         :Refused
    Treatment :
        Convert everything into time per year (day*365) (week*52) ,(month*12)
        Never is mapped to 0
        Don't Know and refused are mapped to the median value



    """

    new_column = column.copy()

    for i in range(len(new_column)):
        if new_column[i] >= 101 and new_column[i] <= 199:
            new_column[i] = (new_column[i] - 100) * 365
        elif new_column[i] >= 201 and new_column[i] <= 299:
            new_column[i] = (new_column[i] - 200) * 52
        elif new_column[i] >= 301 and new_column[i] <= 399:
            new_column[i] = (new_column[i] - 300) * 12
        elif new_column[i] >= 401 and new_column[i] <= 499:
            new_column[i] = new_column[i] - 400
        elif (new_column[i] == 300) or (new_column[i] == 555):
            new_column[i] = 0
    

    filtered_elements = [x for x in new_column if 0 <= x <= 99]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 777) or (new_column[i] == 999) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_12(column):
    """
    Preprocess "Times seen health professional for diabetes"

    Possibles values :
        1-76 : Number of times
        88 : None
        98 : Never heard of
        77 : Don't Know / Not sure
        99 : Refused
    Treatment
        None and Never heard of are mapped to 0 (if the have never heard of it , it means they've never done it )

        Don't Know , refued and Nan are mapped to the median

    """
    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 88 or new_column[i] == 98:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 76]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_13(column):
    """
    Preprocess "A lot , a little , never " kind of data, give them a meaningfull Ordering relation


    Possibles valus
        1 : A lot
        2 : a little
        3 : not at all
        7 Don't know
        9 Not sure
    Treatment

        A lot is mapped to 2
        A little is mapped to 1
        not at all is mapped to 0
        Creates an ordering relation

        Don't Know , not sure and Nan are mapped to the median

    """
    new_column = column.copy()

    new_column[new_column == 3] = 0
    new_column[new_column == 2] = 1
    new_column[new_column == 1] = 2

    filtered_elements = [x for x in new_column if 0 <= x <= 2]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_14(column):
    """
    Preprocess the Join Pain data

    Possibles values :
        0-10 : pain values
        77 Don't know / not sure
        99 Refused
    Treatment
        map Don't known,  refused and Nan to the median vaues



    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 10]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i])
        ):  # DON'T KNOW / REFUSED
            new_column[i] = median

    return new_column


def process_15(column):
    """
    Preprocess Time related question , give them a meaningfull Ordering

    Possibles Values :
        1 : Within the past month
        2 : Withing the past year
        3 : Within the past 2 years
        4 : 2 or more years ago
        7 : Don't know or not sure
        8 : Never
        9 : Refused

    Treatment :
        Never is mapped to 5 ,
        Don't know , not sure and Nan are mapped to the median value

    """
    new_column = column.copy()
    new_column[new_column == 8] = 5

    filtered_elements = [x for x in new_column if 0 <= x <= 5]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_16(column):
    """
    Preprocess "How often" type questions

    Possibles Values :
        1 : Always
        2 : Nearly Always
        3 : Someties
        4 : Seldom
        5 : Never
        7 : DOnDon't know
        8 : not applicable
        9 Refused
    Treatmeant
        We map "not applicable" to 1
        Don't knoww , refused and Nan are mapped to the median value


    """
    new_column = column.copy()

    new_column[new_column == 8] = 1

    filtered_elements = [x for x in new_column if 0 <= x <= 6]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i])
        ):  # DON'T KNOW / REFUSED
            new_column[i] = median

    return new_column


def process_17(column):
    """
    Preprocess data from Cigarette consumption


    Possibles values :
        1-7 : last time cigrattes (increasing order)
        8   : Never
        77  : Don't know
        99  : refused
    Treatment
        Map Don't Know and refused to the median
        (Never is already correclty ordered)

    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 8]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i])
        ):  # DON'T KNOW / REFUSED
            new_column[i] = median

    return new_column


def process_18(column):
    """
    Preprocess ASTHMAGE collumn

    Possibles Values :
        11-97 : 11 or older
        97 : Age 10 or younger
        98 Don't know / not sure
        99 Refused
    Treatment
        Map "age 10 or younger" to 6
        map don't know, refused / nan to the median values


    """
    new_column = column.copy()
    new_column[new_column == 97] = 6

    new_column = column.copy()

    filtered_elements = [x for x in new_column if 11 <= x <= 96]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 98) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_19(column):
    """
    Preprocess Hour/per week data

    Possibles Values
        1-96 : hours
        97 : Don't know / not sure
        98 : Zero
        99 : Refused
    Treatment
        98 maped to 0
        map don't know, refused and Nan to the median values

    """

    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 98:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 96]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 97) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_20(column):
    """
    Preprocess "days over the last 2 weeks" data

    Possibles Values :
        1 -14 : Number of days
        88 : None
        77 DOn't know/not sure
        99 : Refused
    Treatment

        Map None to 0
        map don't know, refused and Nan to the median values

    """

    new_column = column.copy()
    for i in range(len(new_column)):
        if new_column[i] == 88:
            new_column[i] = 0

    filtered_elements = [x for x in new_column if 0 <= x <= 14]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (new_column[i] == 77) or (new_column[i] == 99) or np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_21(column):
    """
    Preprocess "Times per day" data

    Possibles Values :
        0-9999 : times per day
    Treatment

        map Nan to the median values

    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 99999]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_22(column):
    """
    Preprocess Met values related data

    Possibles Values :
        0-128 :  Met value
    Treatment

        Map Nan to the median values

    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 128]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if np.isnan(new_column[i]):
            new_column[i] = median

    return new_column


def process_23(column):
    """
    Preprocess "Max VO2" data

    Possibles Values :
        0-98999 : values
        99900 ,99000 :Don’t know/Not Sure/Refused/Missing
    Treatment

        Map None to 0
        map don't know, refused and Nan to the median values

    """

    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 98999]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if (
            (new_column[i] == 99900)
            or (new_column[i] == 99000)
            or np.isnan(new_column[i])
        ):
            new_column[i] = median

    return new_column


def process_24(column):
    """
    Preprocess BMI DATA

    Possibles Values :
        0-9999 : Bmi
    Treatment
        map Nan to the median values

    """

    new_column = column.copy()

    filtered_elements = [x for x in new_column if 0 <= x <= 9999]
    median = np.median(filtered_elements)

    for i in range(len(new_column)):
        if np.isnan(new_column[i]):
            new_column[i] = median

    return new_column

def process_25(column):
    
    """
    Process for BPHIGH4, 
    We consider 0 : don't have high blood pressure 
    we consider 1 : you have high blood pressure
    we consider 2 : to have borderline high which is even more than high blood pressure 
    - If told yes only during pregnancy that means no since it's normal 
   Same process but with DIABETE3, PREDIAB1, which corresponds to diabetes
   
   """
    new_column = column.copy()
    new_column[new_column==2]=0
    new_column[new_column==3]=0
    new_column[new_column==4]=2
            
    filtered_elements = [x for x in new_column if 0 <= x <= 2]        
    median = np.median(filtered_elements)
        
    for i in range(len(new_column)):
        if ((new_column[i] == 7)  or (new_column[i] == 9) or np.isnan(new_column[i])):
            new_column[i] = median
            
    return new_column


def process_26(column):
    
    """
    Process for PERSDOC2, 
    We consider 0 : don't have a care provider, so 3 becomes 0 
    we consider 1 : you one
    we consider 2 : you have two
   """
    
    new_column = column.copy()

    new_column[new_column==3]=0

    filtered_elements = [x for x in new_column if 0 <= x <= 6]
    median = np.median(filtered_elements)
        
    for i in range(len(new_column)):
        if ((new_column[i] == 7) or (new_column[i] == 9) or np.isnan(new_column[i])): # DON'T KNOW / REFUSED 
            new_column[i] = median
            
    return new_column


def process_cont(column):
    """
    Preprocess continues data 
    
    """
    new_column = column.copy()

    filtered_elements = [x for x in new_column if not np.isnan(x)]
    median = np.median(filtered_elements)
    for i in range(len(new_column)):
        if (np.isnan(new_column[i])):
            new_column[i] = median
    return new_column