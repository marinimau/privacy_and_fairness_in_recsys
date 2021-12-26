"""
PerBlur is extension of previous work proposed by Windenberg et al., (BlurMe: Inferring and Obfuscating
User Gender Based on Ratings ) and Strucks et al., (BlurM(or)e: Revisiting Gender Obfuscation
in the User-Item Matrix)

This code is extending previous github repository done by Christopher Strucks (Github Link: https://github.com/STrucks/BlurMore)

In PerBlur you need to :
    + Generate json file: "Confidence score" from imputation/knn/few_observed_entries
    + You will read the json file
"""

import numpy as np
import matplotlib.pyplot as plt
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def blurMe_1m():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = -1
    p = 0.05
    notice_factor = 2
     # load_user_item_matrix_1m_all
    X = MD.load_user_item_matrix_1m_trainingSet()  # load_user_item_matrix_1m_trainingSet MD.load_user_item_matrix_1m_all()  # max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X_test = MD.load_user_item_matrix_1m_testSet()

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor

    print("obfuscation")
    # Now, where we have the two lists, we can start obfuscating the data:
    # X = MD.load_user_item_matrix_1m()
    X_obf = np.copy(X)

    # X = Utils.normalize(X)
    # X_obf = Utils.normalize(X_obf)
    prob_m = []  # [p / sum(C_m) for p in C_m]
    prob_f = []  # [p / sum(C_f) for p in C_f]
    print("obfuscation")
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        if T[index] == 1:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:  # and X_test [index, int(movie_id) - 1] ==0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int((movie_id) - 1)]  # avg_ratings[int(index)]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100

                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:  # and X_test [index, int(movie_id) - 1] ==0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int((movie_id) - 1)]  # int(index)
                    added += 1
                safety_counter += 1

    # row_sums = X.sum(axis=1)
    # np.save('test_X_obf.npy', X_obf)
    # X_obfu = X_obf * row_sums[:, np.newaxis]

    # X_obfu = Utils.denormalize(XX, X_obf)
    # X_obfu = scaler.inverse_transform(XX)

    # print(X_obfu)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml1m/BlurMe/DoubleCount_BlurMe/"  # "ml-1m/BlurMe/"
        with open(output_file + "TrainingSet_blurMe_ML1M_obfuscated_" + str(
                p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        output_file = "Flixster/DoubleCount_BlurMe/"  # "Flixster/BlurMe/" FX/
        with open(output_file + "TrainingSet_FX_DCount_excludeTestSet_blurme_obfuscated_" + str(
                p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
            top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'LFM':
        output_file = "lastFM/BlurMe/DoubleCount_BlurMe/"  # "Flixster/BlurMe/"
        with open(output_file + "TrainingSet_DCount_excludeTestSet_LFM_blurme_obfuscated_" + str(
                p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
            top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


def rating_add_1m():
    # add a percentage of random ratings to a user:
    X = MD.load_user_item_matrix_1m()
    X_obf = MD.load_user_item_matrix_1m()
    percentage = 0.05
    for user_index, user in enumerate(X):
        nr_ratings = 0
        for rating in user:
            if rating > 0:
                nr_ratings += 1

        added = 0
        safety_counter = 0
        while added < nr_ratings * percentage and safety_counter < 100:
            index = np.random.randint(0, len(user))
            if X_obf[user_index, index] > 0:
                safety_counter += 1
                continue
            else:
                X_obf[user_index, index] = np.random.randint(1, 6)

    # output the data in a file:
    with open("ml-1m/random_added_obfuscated_" + str(percentage) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                        int(rating)) + "::000000000\n")
    return X_obf


def rating_swap_1m():
    plot = False
    low_bound, high_bound = 100, 1500
    # swap 0 ratings with non zero ratings:
    X = np.transpose(MD.load_user_item_matrix_1m())
    X_obf = np.transpose(MD.load_user_item_matrix_1m())
    nr_ratings = []
    for item in X:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    if plot:
        # plt.subplot(1,2,1)
        ax1.bar(range(1, len(X) + 1), nr_ratings)
        ax1.set_xlabel("movie id")
        ax1.set_ylabel("nr ratings")

    # we want to remove ratings from movies that have more than 1500 ratings:
    amount_removed = 0
    for item_index, item in enumerate(X):
        if nr_ratings[item_index] > high_bound:
            indecies = np.argwhere(X[item_index, :] > 0)[:, 0]
            indecies = np.random.choice(indecies, size=(nr_ratings[item_index] - high_bound,), replace=False)
            amount_removed += len(indecies)
            for i in indecies:
                X_obf[item_index, i] = 0
    """ To check if the removal is working

    nr_ratings = []
    for item in X_obf:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)
    if plot:
        plt.bar(range(1, len(X) + 1), nr_ratings)
        plt.xlabel("movie id")
        plt.ylabel("nr ratings")
        plt.show()

    """
    # now we want to add ratings to movies with a small number of ratings:
    print(np.asarray(nr_ratings))
    indecies = np.argwhere(np.asarray(nr_ratings) < low_bound)[:, 0]
    print(indecies)
    nr_few_rated_movies = len(indecies)
    nr_to_be_added = amount_removed / nr_few_rated_movies
    print(nr_to_be_added)
    for item_index, item in enumerate(X):
        if nr_ratings[item_index] < low_bound:
            indecies = np.argwhere(X[item_index, :] == 0)[:, 0]
            indecies = np.random.choice(indecies, size=(int(nr_to_be_added),), replace=False)
            for i in indecies:
                X_obf[item_index, i] = np.random.randint(1, 6)

    """ To check if the removal and adding is working
    """
    nr_ratings = []
    for item in X_obf:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)
    if plot:
        # plt.subplot(1,2,2)
        ax2.bar(range(1, len(X) + 1), nr_ratings)
        ax2.set_xlabel("movie id")
        ax2.set_ylabel("nr ratings")
        plt.show()

    X_obf = np.transpose(X_obf)

    # output the data in a file:
    with open("ml-1m/rebalanced_(" + str(low_bound) + "," + str(high_bound) + ").dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                        int(rating)) + "::000000000\n")

    return X_obf


def blurMePP():
    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    removal_mode = list(['random', 'strategic'])[1]
    rating_mode = list(['avg', 'predicted'])[0]
    # id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.05
    dataset = ['ML', 'Fx', 'LFM', 'Li'][2]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m_all()
        # X = MD.load_user_item_matrix_1m_trainingSet()  # load_user_item_matrix_1m_trainingSet max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
        X_test = MD.load_user_item_matrix_1m_testSet()
        # X = MD.load_user_item_matrix_100k()
        # T = MD.load_gender_vector_100k()
    elif dataset == 'Fx':
        """import FlixsterData as FD
        #X, T, _ = FD.load_flixster_data_subset()
        X, T, _ = FD.load_flixster_data_subset_trainingSet()"""
        import FlixsterDataSub as FDS
        # X = FDS.load_user_item_matrix_FX_All()
        X = FDS.load_user_item_matrix_FX_TrainingSet()
        X_test = FDS.load_user_item_matrix_FX_Test()
        T = FDS.load_gender_vector_FX()
    elif dataset == 'LFM':
        import LastFMData as LFM
        # X = LFM.load_user_item_matrix_lfm_Train()  # LFM.load_user_item_matrix_lfm_All()
        X = LFM.load_user_item_matrix_lfm_All()  # load_user_item_matrix_lfm_Train LFM.load_user_item_matrix_lfm_All()
        T = LFM.load_gender_vector_lfm()
        X_test = LFM.load_user_item_matrix_lfm_Test()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    # X = Utils.normalizze(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    """from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(x[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))


    if top == -1:
        values = coefs[:,2]
        index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1][100:]
        # print(len(L_m))
        R_m = 2835 - coefs[:top_male, 0] #3952 2835
        C_m = np.abs(coefs[:top_male, 2])
        # C_m = [x for x in C_m if x > 2] # C_m[C_m <=  2]
        # print("C_m", type (C_m), "\n", C_m)
        L_f = coefs[coefs.shape[0] - top_female:, 1][100:]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))
        # C_f = [x for x in C_f if x > 2] # C_f[C_f <= 2]
        # print("C_f", type (C_f), "\n", C_f)

        # plt.plot(C_m, label = 'Male Coef', c= 'lightskyblue')
        # plt.plot(C_f, label = 'Female Coef', c= 'lightpink')
        # plt.axhline(y=2, color='crimson', linestyle='--')
        # plt.legend(loc="upper right")
        # plt.title("Male and Female coefficients on Flixster Data", fontsize=16, fontweight="bold")
        # plt.xlabel ('Features', fontsize=19)
        # plt.ylabel ('Coefficients', fontsize=19)
        # # plt.savefig("threshold_ML1M_IndicativeItems.pdf")
        # plt.show()

    else:
        L_m = coefs[:top, 1]
        R_m = 2835 -coefs[:top, 0] #3952 2835
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0]-top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0]-top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0]-top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # print(len(L_f))
    # Here we are trying to get all the less indicative items for F / M
    # Based on the plot we see that from 600 to the end the coefficients are <= 2
    L_ff = L_f.copy()
    print(L_ff)
    ## low indicative items
    # L_ff = L_ff [100:]
    # highly indicative items
    # L_ff = L_ff[:400]
    # print("L_ff:", L_ff, "\n\n", len(L_ff))
    L_ff = pd.DataFrame(L_ff)
    L_ff.to_csv('L_f_FX_Normalized.csv', index=False)
    # print("------")
    # print(len(L_m))
    L_mm = L_m.copy()
    print (L_mm)
    ## low indicative items
    # L_mm = L_mm [100:]
    # highly indicative items
    # L_mm = L_mm[:400]
    # print("L_mm:", L_mm, "\n\n", len( L_mm))
    L_mm = pd.DataFrame(L_mm)
    L_mm.to_csv('L_m_FX_Normalized.csv', index=False)"""
    # Now, where we have the two lists, we can start obfuscating the data:
    # X = MD.load_user_item_matrix_1m()
    # np.random.shuffle(X)
    # print(X.shape)

    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index_m = 0
        greedy_index_f = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index_m >= len(L_m):
                    safety_counter = 100
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                greedy_index_m += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:  # and X_test [index, int(movie_id) - 1] ==0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index_f >= len(L_f):
                    safety_counter = 100
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                greedy_index_f += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:  # and X_test [index, int(movie_id) - 1] ==0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    if removal_mode == "random":
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:  # 200 for ML1M and 300 for Flixster
                nr_many_ratings += 1
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                          size=(int(nr_remove),))  # ,replace=False)
                X_obf[user_index, to_be_removed_indecies] = 0
    else:
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        print("nr_many_ratings:", nr_many_ratings)
        print("total_added:", total_added)
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            print("user: ", user_index)
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                index_m = 0
                index_f = 0
                rem = 0
                if T[user_index] == 1:
                    safety_counter = 0
                    # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                    while (rem < nr_remove) and safety_counter < 100:
                        if index_f >= len(L_f):
                            safety_counter = 100
                            continue

                        if removal_mode == "random":
                            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                      size=(int(nr_remove),),
                                                                      replace=False)  # , replace=False)
                        if removal_mode == "strategic":
                            to_be_removed_indecies = L_f[index_f]
                        index_f += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1

                elif T[user_index] == 0:

                    while (rem < nr_remove) and safety_counter < 100:
                        if index_m >= len(L_m):  # and safety_counter < 1000:
                            safety_counter = 100
                            continue

                        if removal_mode == "random":
                            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                      size=(int(nr_remove),),
                                                                      replace=False)  # , replace=False)
                        # X_obf[user_index, to_be_removed_indecies] = 0

                        if removal_mode == "strategic":
                            to_be_removed_indecies = L_m[index_m]
                        index_m += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1

    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml1m/"  # "ml-1m/BlurMore/" ml-1m/BlurMore/Random_Removal/
        with open(output_file + "All_testSafe`Count_threshold20_ML1M_blurmepp_obfuscated_" + sample_mode + "_" +
                  str(p) + "_" + str(notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:  # + "_" + str(removal_mode) + ".dat",
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        output_file = "Flixster/"  # BlurMore/RandomRem/" # "Flixster/BlurMore/Greedy_Removal/" FX/
        with open(
                output_file + "All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_" + sample_mode + "_" + str(
                        p) + "_" + str(
                        notice_factor) + "_" + str(removal_mode) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'LFM':
        output_file = "lastFM/"  # BlurMore/RandomRem/"
        with open(output_file + "All_testSafe`Count_LFM_blurmepp_ExcludeTestSet_obfuscated_" + sample_mode + "_" + str(
                p) + "_" + str(notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


# ---------------------------------
"""Creation of Personalized list of indicative items """


def Personalized_list_User():
    notice_factor = 2
    item_choice = {}
    # lastFM/NN_LFM_TrainingSet_allUsers_KNN_fancy_imputation
    # lastFM/NN_LFM_All_allUsers_KNN_fancy_imputation
    with open(
            'ml-1m/user_based_imputation/With_Fancy_KNN/test_Confidence_Score_Items_Selection/NN_All_AllUsers_Neighbors_Weight_K_30_item_choice.json') as json_file:
        data = json.load(
            json_file)
    len_dict = {}
    for key, value in data.items():
        # print (value)
        length = []
        for v in value:
            # print (len(v))
            length.append(len(v))
        len_dict[int(key)] = length

    dataset = ['ML', 'Fx', 'LFM', 'Li'][0]
    if dataset == 'ML':  #
        # X = MD.load_user_item_matrix_1m_trainingSet()  #
        X = MD.load_user_item_matrix_1m_all()
        T = MD.load_gender_vector_1m()  #
    elif dataset == 'Fx':
        import FlixsterDataSub as FDS
        X = FDS.load_user_item_matrix_FX_All()
        # X = FDS.load_user_item_matrix_FX_TrainingSet()
        T = FDS.load_gender_vector_FX()
    elif dataset == 'LFM':
        import LastFMData as LFM
        X = LFM.load_user_item_matrix_lfm_All()
        # X = LFM.load_user_item_matrix_lfm_Train()
        T = LFM.load_gender_vector_lfm()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    k = 100
    L_mm = list(map(int, L_m))
    L_mm = list(map(lambda x: x - 1, L_mm))[:k]
    L_ff = list(map(int, L_f))
    L_ff = list(map(lambda x: x - 1, L_ff))[:k]

    for z in range(len(X)):
        print(z)
        values = len_dict[z]
        lst_j = []
        # list of neighbors ordered / ranked by weight for user i
        user_item = list(np.argsort(values))  # [::-1])
        # lst = X_filled [z]
        # lst = list(map(lst.__getitem__, user_item))
        if (len(user_item) == len(values)):
            p = 0
            while p < len(values):
                if T[z] == 0:
                    f = user_item.pop(0)  # np.argmin (lst)
                    if f in L_ff:
                        if f not in lst_j:
                            lst_j.append(f)

                elif T[z] == 1:
                    m = user_item.pop(0)
                    if m in L_mm:
                        if m not in lst_j:
                            lst_j.append(m)
                p += 1
            item_choice[z] = lst_j
    print("item_choice: ", item_choice)
    # lastFM/With_Fancy_KNN/lfm_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top500IndicativeItems.json
    with open(
            "ml1m/With_Fancy_KNN/ML1M_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top100IndicativeItems_noRemoval.json",
            "w") as fp:
        json.dump(item_choice, fp, cls=NpEncoder)


# --------------------------------

"""PerBlur without removal strategy function for obfuscating the user-item matrix"""


def PerBlur_No_Removal():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = -1
    p = 0.05
    notice_factor = 2
    dataset = ['ML', 'Fx', 'LFM', 'Li'][0]
    if dataset == 'ML':  # load_user_item_matrix_1m_all
        X = MD.load_user_item_matrix_1m_all()  # load_user_item_matrix_1m_trainingSet
        # X = MD.load_user_item_matrix_1m_trainingSet()
        T = MD.load_gender_vector_1m()  #
        X_test = MD.load_user_item_matrix_1m_testSet()
        X_filled = MD.load_user_item_matrix_1m_complet()
    elif dataset == 'Fx':

        import FlixsterDataSub as FDS
        X = FDS.load_user_item_matrix_FX_All()
        # X = FDS.load_user_item_matrix_FX_TrainingSet()
        X_test = FDS.load_user_item_matrix_FX_Test()
        T = FDS.load_gender_vector_FX()
        X_filled = FDS.load_user_item_FX_Complet()
    elif dataset == 'LFM':
        import LastFMData as LFM
        # X = LFM.load_user_item_matrix_lfm_Train()  # LFM.load_user_item_matrix_lfm_All()
        X = LFM.load_user_item_matrix_lfm_All()  # load_user_item_matrix_lfm_Train LFM.load_user_item_matrix_lfm_All()
        T = LFM.load_gender_vector_lfm()
        X_test = LFM.load_user_item_matrix_lfm_Test()
        # X_filled = LFM.load_user_item_matrix_lfm_complet()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    # lastFM/With_Fancy_KNN/lfm_NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_TopAllIndicativeItems.json  lfm_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems
    # ml1m/ML1M_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems
    # Flixster/With_Fancy_KNN/FX_NN_TrainingSet_2370_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json  FX_NN_All_2370_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems
    # lfm_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top1000IndicativeItems
    # ml-1m/PerBlur/test_Confidence_Score_Items_Selection/NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_Top100IndicativeItems.json
    with open(
            'ml-1m/PerBlur/test_Confidence_Score_Items_Selection/NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_Top100IndicativeItems.json') as json_file:

        item_choice = json.load(json_file)

    # Now, where we have the two lists, we can start obfuscating the data:
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        added = 0
        mylist = list(item_choice.values())
        safety_counter = 0

        while added < k and safety_counter < 100:  # 1000
            if greedy_index >= len(mylist[index]):
                safety_counter = 100
                continue
            if sample_mode == 'greedy':
                vec = mylist[index]
                movie_id = vec[greedy_index]
            if sample_mode == 'random':
                movie_id = vec[np.random.randint(0, len(vec))]
            greedy_index += 1
            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, movie_id]])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:  # and X_test [index, int(movie_id) ] == 0:
                X_obf[index, movie_id] = avg_ratings[
                    int(movie_id)]  # X_filled[index, movie_id] # avg_ratings[int(movie_id)] #
                added += 1
            safety_counter += 1
        total_added += added

    # output the data in a file:

    output_file = ""
    if dataset == 'ML':
        output_file = "ml1m/PerBlur/Top-100-NoRemoval/"  # ml1m/
        with open(
                output_file + "All_thresh20_PerBlur_ML1M_obfuscated_Top100IndicativeItems_" + rating_mode + "_" + sample_mode + "_" + str(
                        p) + "_" + str(notice_factor) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")


    elif dataset == 'Fx':
        output_file = "Flixster/BlurSome/Top-100-NoRemoval/"  # "Flixster/BlurSome/Top-100/" FX
        with open(
                output_file + "All_thresh20_NoRemoval_blurSome_FX_obfuscated_Top100IndicativeItems_" + rating_mode + "_" + sample_mode + "_" + str(
                        p) + "_" + str(
                        notice_factor) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'LFM':
        output_file = "lastFM/PerBlur/Top-All-NoRemoval/"
        with open(
                output_file + "All_LFM_NoRemoval_PerBlur_obfuscated_TopAll_" + rating_mode + "_" + sample_mode + "_" + str(
                        p) +
                "_" + str(notice_factor) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


# -----------------------------------------------

"""PerBlur with removal function for obfuscatibg the user-item matrix"""


def PerBlur():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    removal_mode = list(['random', 'strategic'])[0]
    top = -1
    p = 0.05
    notice_factor = 2
    dataset = ['ML', 'Fx', 'LFM', 'Li'][2]
    if dataset == 'ML':  # load_user_item_matrix_1m_all
        X = MD.load_user_item_matrix_1m_trainingSet()  #
        T = MD.load_gender_vector_1m()  #
        X_test = MD.load_user_item_matrix_1m_testSet()
        X_filled = MD.load_user_item_matrix_1m_complet()
    elif dataset == 'Fx':

        import FlixsterDataSub as FDS
        # X = FDS.load_user_item_matrix_FX_All()
        X = FDS.load_user_item_matrix_FX_TrainingSet()
        X_test = FDS.load_user_item_matrix_FX_Test()
        T = FDS.load_gender_vector_FX()
        X_filled = FDS.load_user_item_FX_Complet()
    elif dataset == 'LFM':
        import LastFMData as LFM
        # X = LFM.load_user_item_matrix_lfm_Train()  # LFM.load_user_item_matrix_lfm_All()
        X = LFM.load_user_item_matrix_lfm_All()  # load_user_item_matrix_lfm_Train LFM.load_user_item_matrix_lfm_All()
        T = LFM.load_gender_vector_lfm()
        X_test = LFM.load_user_item_matrix_lfm_Test()
        X_filled = LFM.load_user_item_matrix_lfm_complet()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    # lastFM/With_Fancy_KNN/lfm_NN_All_AllUsers_Neighbors_Weight_K_30_item_choice_TopAllIndicativeItems.json
    # ml1m/ML1M_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json
    # Flixster/With_Fancy_KNN/FX_NN_TrainingSet_2370_AllUsers_Neighbors_Weight_K_30_item_choice_Top50IndicativeItems.json
    with open(
            'lastFM/With_Fancy_KNN/lfm_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top500IndicativeItems.json') as json_file:

        item_choice = json.load(json_file)

    # Now, where we have the two lists, we can start obfuscating the data:
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        added = 0
        mylist = list(item_choice.values())
        safety_counter = 0

        while added < k and safety_counter < 100:
            if greedy_index >= len(mylist[index]):
                safety_counter = 100
                continue
            if sample_mode == 'greedy':
                vec = mylist[index]
                movie_id = vec[greedy_index]
            if sample_mode == 'random':
                movie_id = vec[np.random.randint(0, len(vec))]
            greedy_index += 1
            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, movie_id]])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:  # and X_test [index, int(movie_id) ] == 0:
                X_obf[index, movie_id] = avg_ratings[
                    int(movie_id)]  # X_filled[index, movie_id] # avg_ratings[int(movie_id)] #
                added += 1
            safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    if removal_mode == "strategic":
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        print("nbr user with profile length > 20: ", nr_many_ratings)
        print("total_added: ", total_added)
        nr_remove = total_added / nr_many_ratings
        print("nr_remove: ", nr_remove)

        for user_index, user in enumerate(X):
            print("user: ", user_index)
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                index_m = 0
                index_f = 0
                rem = 0
                if T[user_index] == 1:
                    safety_counter = 0
                    # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                    while (rem < nr_remove) and safety_counter < 100:
                        if index_f >= len(L_f):  # and safety_counter < 1000:
                            safety_counter = 100
                            continue

                        to_be_removed_indecies = L_f[index_f]
                        index_f += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1

                elif T[user_index] == 0:

                    while (rem < nr_remove) and safety_counter < 100:
                        if index_m >= len(L_m):  # and safety_counter < 1000:
                            safety_counter = 100
                            continue

                        to_be_removed_indecies = L_m[index_m]
                        index_m += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1
    else:
        # Now remove ratings from users that have more than 200 ratings equally:

        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                          size=(int(nr_remove),))  # replace=False
                X_obf[user_index, to_be_removed_indecies] = 0

    # output the data in a file:

    output_file = ""
    if dataset == 'ML':
        output_file = "ml1m/PerBlur/Top-50/"  # ml1m/
        with open(
                output_file + "TrainingSet_thresh20_PerBlur_ML1M_obfuscated_Top50IndicativeItems_" + rating_mode + "_" + sample_mode + "_" + str(
                        p) + "_" + str(notice_factor) + "_" + str(removal_mode) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")


    elif dataset == 'Fx':
        output_file = "Flixster/BlurSome/Top-50-ExcludeTest/"  # "Flixster/BlurSome/Top-100/" FX
        with open(
                output_file + "TrainingSet_thresh20_blurSome_FX_obfuscated_Top50IndicativeItems_" + rating_mode + "_" + sample_mode + "_" + str(
                        p) + "_" + str(
                        notice_factor) + "_" + str(removal_mode) + ".dat",
                'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'LFM':
        output_file = "lastFM/PerBlur/Top-500-RightSplit/"
        with open(output_file + "All_LFM_PerBlur_obfuscated_Top500_" + rating_mode + "_" + sample_mode + "_" + str(
                p) + "_" + str(notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf


# blurMe_100k()
# blurMe_1m()
blurMePP()
# Personalized_list_User() # This will create the lists of indicative items. It goes before the PerBlur function
# PerBlur ()
# PerBlur_No_Removal ()
