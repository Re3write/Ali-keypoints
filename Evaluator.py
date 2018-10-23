import csv
import math
import numpy as np

class FaiKeypoint2018Evaluator():
    def __init__(self,userAnswerFile = None, standardAnswerFile = None):
        self.user_answer_file = userAnswerFile
        self.standard_answer_file=standardAnswerFile
        self.user_answer_dict = {}
        self.standard_answer_dict = {}
        self.top_error=[]
        with open(self.user_answer_file, "r") as f:
            reader = csv.reader(f)
            # next(reader)
            for value in reader:
                joint_ = []
                if len(value) == 0:
                    break
                name = value[0]
                self.user_answer_dict[name] = value

            # print(self.user_answer_dict)
        with open(self.standard_answer_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for value in reader:
                joint_ = []
                if len(value) == 0:
                    break
                name = value[0]
                self.standard_answer_dict[name] = value

    def evaluate(self):
        count = 0
        error_sum = .0
        for name in self.standard_answer_dict:
            print(self.standard_answer_dict[name])
            standard_answer = self.standard_answer_dict[name]
            standardKps = self.parseKeypoint(standard_answer)
            if standardKps==None:
                continue
            if standard_answer[1]=='skirt' or standard_answer[1] == 'trousers':
                normalize_factor = self.calcDistance(standardKps[15],standardKps[16])
            else:
                normalize_factor = self.calcDistance(standardKps[5], standardKps[6])

            if normalize_factor < 10:
                continue

            this_count = 0
            for i in range(len(standardKps)):
                if standardKps[i][2] == 1:
                    this_count += 1

            count += this_count
            user_answer=None
            user_answer = self.user_answer_dict[name]
            if user_answer == None or len(user_answer) != len(standard_answer):
                error_sum += this_count
                per_pic_error = this_count
                per_pic = (name, per_pic_error)
                self.top_error.append(per_pic)
                continue

            userKps = self.parseKeypoint(user_answer)

            print("error:",name,error_sum)
            per_pic_error = self.calcError(standardKps, userKps)/normalize_factor
            error_sum += per_pic_error
            per_pic=(name,per_pic_error)
            self.top_error.append(per_pic)

        score = error_sum / count
        return score



    def parseKeypoint(self, key_points):
        new_key_points = []
        for i in range(2,len(key_points)):
            x,y,visible = key_points[i].split('_')
            new_key_points.append([int(x), int(y), int(visible)])
        # print("new_key_points:",new_key_points)
        return new_key_points

    def calcDistance(self, a, b):
        # print("a",a)
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

    def calcError(self, standard, user):
        error = .0
        for i in range(len(standard)):
            if standard[i][2] == 1:
                error += self.calcDistance(standard[i], user[i])
        return error

    def writerror(self, result_path):

        with open(result_path, 'w', newline='') as f:
            # json.dump(dump_results, f)
            top_error=sorted(self.top_error,key=lambda error: error[1],reverse=True)

            writer = csv.writer(f)
            writer.writerows(top_error)




if __name__=="__main__":
    Evaluator = FaiKeypoint2018Evaluator(userAnswerFile="./logs/results1017_12.csv",
                                         standardAnswerFile="fashionAI_key_points_test_a_answer_20180426.csv")
    score = Evaluator.evaluate()

    Evaluator.writerror(result_path="./logs/toperror1.csv")

    print("score",score)