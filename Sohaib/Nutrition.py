import csv
from collections import OrderedDict
import re


class Nutrition:
    def __init__(self,guideline_file='guidelines.csv', nutrition_file='fruits.csv'):
        self.guidelines = self.loadGuidelines(guideline_file)
        self.all_nutrition = self.loadNutritionalData(nutrition_file)


    def getFoodAmount(self,gender='F', age='27', class_name='Apple'):
        return self.getMaxAllowed(self.getDietaryGuidelines(self.guidelines, gender, age), self.all_nutrition[class_name])

    def getDietaryGuidelines(self, guidelines, gender='M', age=2):
        if float(age)>=2. and (gender == 'M' or gender == 'F'):
            guide = guidelines[gender]
            ages = list(guide.keys())
            for key_age in ages:
                if float(age)>=float(key_age):
                    break
            return guide[key_age]

    def loadGuidelines(self, filename = 'guidelines.csv'):
        with open(filename) as f:
            guidelines = [{k: v for k, v in row.items()}
                          for row in csv.DictReader(f, skipinitialspace=True)]
            output_guidelines = {'F': OrderedDict(),
                                 'M': OrderedDict()
                                 }

            for row in guidelines:
                gender = row.pop('Gender')
                lower_limit = row.pop('Lower_Limit')
                higher_limit = row.pop('Higher_Limit')
                output_guidelines[gender][lower_limit] = row
            return output_guidelines

    def getMaxAllowed(self,person_max, food_nutrients):
        allowed = []
        for nutrient, allowed_value in person_max.items():
            allowed_value = float(allowed_value.replace(',',''))
            if nutrient in food_nutrients:
                try:

                    if nutrient in food_nutrients:
                        nutrient_in_food = self.strToFloat(food_nutrients[nutrient])
                        # nutrient_in_food = float(food_nutrients[nutrient].replace(',',''))
                        if nutrient_in_food > 0.0:
                            allowed.append(allowed_value/self.strToFloat(nutrient_in_food))
                except:
                    print(nutrient)
                    print(food_nutrients[nutrient])


        return min(allowed)*100

    def loadNutritionalData(self, filename = 'fruits.csv'):
        with open(filename) as f:
            fruits = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]
            nutrients = {}
            for row in fruits:
                nutrients[row['name']] = row
            return nutrients

    def strToFloat(self, str):

        if type(str) == type(''):
            if str == '':
                return 0.0
            return float(re.sub(r'[^\d.]+', '', str))
        else:
            return float(str)