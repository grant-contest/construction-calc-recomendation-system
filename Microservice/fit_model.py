import numpy as np
import pandas as pd


data_path = "Data/"

"""# РСХБ"""
import random

regions = pd.read_csv(data_path + "region.csv")
regions = regions[regions.columns[2]]
b = pd.Series(["Донецкая Народная Республика (ДНР)", "Луганская Народная Республика (ЛНР)", "Запорожская область", "Херсонская область"], index=[86, 87, 88, 89])
regions = regions._append(b)
regions = regions.drop(85)
regions

n_rows = 1000

def wallsMaterialRestriction (foundation):
  if foundation in {"Свайный", "Столбчатый"}:
    return np.random.choice(["Дерево", "Каркас"])
  return np.random.choice(["Кирпич", "Легкий бетон", "Дерево", "Каркас"])

def facadeTechnologyRestriction (wallsMaterial):
  if wallsMaterial == "Каркас":
    return np.random.choice(["Панели (сайдинг)","Облицовка кирпичом", "Искусственный камень"])
  if wallsMaterial == "Дерево":
    return np.random.choice(["Без отделки", "Панели (сайдинг)"])
  if wallsMaterial == "Кирпич":
    return "Без отделки"
  return np.random.choice(["Без отделки", "Панели (сайдинг)", "Облицовка кирпичом", "Искусственный камень"])

def ladderMaterialRestriction (floorCount, foundationType):
  if floorCount == 1:
    return "-"
  if foundationType in {"Свайный", "Столбчатый"}:
    return np.random.choice(["Дерево", "Металл"])
  return np.random.choice(["-", "Дерево", "Бетон", "Металл"])

def foundationTypeRestriction(floorCount):
  if floorCount == 3:
    return np.random.choice(["Ленточный", "Плитный"])
  return np.random.choice(["Свайный", "Столбчатый", "Ленточный", "Плитный"])

def any_to_0_1(a):
  if a > 0:
    return 1.
  else:
    return 0.

def generate_df(n_rows):
  floorCounts = [(float(random.randint(1, 3))) for i in range(n_rows)]
  foundationTypes = [foundationTypeRestriction(floorCounts[i]) for i in range(n_rows)]
  wallsMaterial = [wallsMaterialRestriction(foundationTypes[i]) for i in range(n_rows)]

  df = pd.DataFrame({
  # step 0
  "houseArea": [float(random.randint(25, 250)) for i in range(n_rows)],
  "siteArea": [float(random.randint(0, 100)) for i in range(n_rows)],
  "floorCount": floorCounts,
  "region": [np.random.choice(regions) for i in range(n_rows)],
  "budgetFloor": [float(random.randint(500_000, 5_000_000)) for i in range(n_rows)],
  "budgetCeil": [float(random.randint(5_000_000, 25_000_000)) for i in range(n_rows)],

  # step 1
  # sitePreparation
  "siteChoosing": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "geologicalWorks": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "geodeticalWorks": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "cuttingBushesAndSmallForests": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "clearingTheSiteOfDebris": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # siteWorks
  "cameras": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "temporaryFence": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # houseDesignAndProject
  "homeProject": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "designProject": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # step 2
  "foundationType": foundationTypes,

  # step 3
  "wallsMaterial": wallsMaterial,

  # step 4
  "slopesNumber": [random.randint(1, 4) for i in range(n_rows)],
  "roofType": [np.random.choice(["Металлочерепица", "Гибкая черепица", "Рулонные материалы", "Ондулин", "Профнастил"]) for i in range(n_rows)],

  # step 5
  "facadeTechnology": [facadeTechnologyRestriction(wallsMaterial[i]) for i in range(n_rows)],

  # step 6
  "windowMaterial": [np.random.choice(["Деревянные", "Пластиковые"]) for i in range(n_rows)],
  "windowType": [np.random.choice(["Однокамерные", "Двухкамерные", "Трехкамерные"]) for i in range(n_rows)],
  "doorMaterial": [np.random.choice(["Деревянные", "Пластиковые"]) for i in range(n_rows)],

  # step 7
  # electrician
  "plasticBoxesUpTo40mmWide": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "layingAThreeToFive": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "cableLaying": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfTwoKey": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfSingleKey": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "recessedTypeSocketDevice": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfPendant": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "chandeliersAndPendants": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # waterSupply
  # "layingOfInternalPipelines"
  "layingOfInternalWaterSupplyPipelines": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfBathtubs": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfSingle": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfMixer": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # sewerage
  "installationOfToilet": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "layingOfSewerage50mm": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "layingOfSewerage110mm": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # heating
  "assemblyOfAWaterSupply": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  # "layingOfInternalPipelines"
  "layingOfInternalHeatingPipelines": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "installationOfWindowFixtures": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  # ventilation
  "installationOfSplitSystems": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "cablingOnABrickWall": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

  #step 8
  "warmFloor": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "ladderMaterial": [ladderMaterialRestriction(floorCounts[i], foundationTypes[i]) for i in range(n_rows)],

  #step 9
  "wallDecoration": [np.random.choice(["Декоративная штукатурка", "Покраска", "Обои", "Плитка"]) for i in range(n_rows)],
  "floorCovering": [np.random.choice(["Ламинат", "Линолеум"]) for i in range(n_rows)],
  "ceilCovering": [np.random.choice(["Натяжной потолок", "Окраска", "Обои", "Штукатурка"]) for i in range(n_rows)]
  })
  # restrictions:

  # 1.
  # Step 2 ("Свайный"): -> step 3 ("Дерево" or "Каркас")
  # Step 2 ("Столбчатый"): -> step 3 ("Дерево" or "Каркас")
  # Step 2 ("Ленточный"): -> step 3 (*)
  # Step 2 ("Плитный"): -> step 3 (*)

  # 2.
  # Step 3 ("Каркас"): -> step 5 ("Панели (сайдинг)" or "Облицовка кирпичом" or "Искусственный камень")
  # Step 3 ("Дерево"): -> step 5 ("Без отделки" or "Панели (сайдинг)")
  # Step 3 ("Кирпич"): -> step 5 ("Без отделки")
  # Step 3 ("Легкий бетон"): -> step 5 (*)

  # 3.
  # Step 0 ("floorCount" == 1) -> step 8 ("ladderMaterial" = "-")

  # 4.
  # Step 0 ("floorCount" == 3) -> step 2 ("Ленточный" or "Плитный")

  # 5.
  # step 2 ("Свайный") -> step 8 ("ladderMaterial" = {"Дерево", "Металл"})
  # step 2 ("Столбчатый") -> step 8 ("ladderMaterial" = {"Дерево", "Металл"})
  # step 2 ("Ленточный") -> step 8 ("ladderMaterial" = *)
  # step 2 ("Плитный") -> step 8 ("ladderMaterial" = *)

  return df
train_test_quotient = 1
train_df = generate_df(n_rows)
test_df = generate_df(int(n_rows * train_test_quotient))
test_sample = generate_df(1)

# val_df = generate_df(int(n_rows * 0.2 * 0.2))

steps_quantity = 10
steps = [None] * steps_quantity
steps[0] = {"houseArea", "siteArea", "floorCount", "region", "budgetFloor", "budgetCeil"}
steps[1] = {"siteChoosing", "geologicalWorks", "geodeticalWorks", "cuttingBushesAndSmallForests", "clearingTheSiteOfDebris", "cameras", "temporaryFence", "homeProject", "designProject"}
steps[2] = {"foundationType"}
steps[3] = {"wallsMaterial"}
steps[4] = {"slopesNumber", "roofType"}
steps[5] = {"facadeTechnology"}
steps[6] = {"windowMaterial", "windowType", "doorMaterial"}
steps[7] = {"plasticBoxesUpTo40mmWide", "layingAThreeToFive", "cableLaying", "installationOfTwoKey", "installationOfSingleKey",
            "recessedTypeSocketDevice", "installationOfPendant", "chandeliersAndPendants", "layingOfInternalWaterSupplyPipelines",
            "installationOfBathtubs", "installationOfSingle", "installationOfMixer", "installationOfToilet", "layingOfSewerage50mm",
            "layingOfSewerage110mm", "assemblyOfAWaterSupply", "layingOfInternalHeatingPipelines", "installationOfWindowFixtures",
            "installationOfSplitSystems", "cablingOnABrickWall"}
steps[8] = {"warmFloor", "ladderMaterial"}
steps[9] = {"wallDecoration", "floorCovering", "ceilCovering"}




cols_dense = [
  "houseArea",
  "siteArea",
  "floorCount",
  "budgetFloor",
  "budgetCeil"
    ]
cols_sparse = [
    # step 0
    "region",

    # step 1
    # sitePreparation
    "siteChoosing",
    "geologicalWorks",
    "geodeticalWorks",
    "cuttingBushesAndSmallForests",
    "clearingTheSiteOfDebris",
    # siteWorks
    "cameras",
    "temporaryFence",
    # houseDesignAndProject
    "homeProject",
    "designProject",

    # step 2
    "foundationType",

    # step 3
    "wallsMaterial",

    # step 4
    "slopesNumber",
    "roofType",

    # step 5
    "facadeTechnology",

    # step 6
    "windowMaterial",
    "windowType",
    "doorMaterial",

      # step 7
    # electrician
    "plasticBoxesUpTo40mmWide",
    "layingAThreeToFive",
    "cableLaying",
    "installationOfTwoKey",
    "installationOfSingleKey",
    "recessedTypeSocketDevice",
    "installationOfPendant",
    "chandeliersAndPendants",

    # waterSupply
    "layingOfInternalWaterSupplyPipelines",
    "installationOfBathtubs",
    "installationOfSingle",
    "installationOfMixer",

    # sewerage
    "installationOfToilet",
    "layingOfSewerage50mm",
    "layingOfSewerage110mm",

    # heating
    "assemblyOfAWaterSupply",
    "layingOfInternalHeatingPipelines",
    "installationOfWindowFixtures",

    # ventilation
    "installationOfSplitSystems",
    "cablingOnABrickWall",

    #step 8
    "warmFloor",
    "ladderMaterial",

    #step 9
    "wallDecoration",
    "floorCovering",
    "ceilCovering"
    ]

assert len(train_df.columns) == len(cols_dense) + len(cols_sparse)

# len(train_df[cols_sparse])
train_df[cols_dense].head()

train_df[cols_sparse].dtypes

# train_df[cols_sparse]
train_df.dtypes
# map_sparse = {}
# map_sparse_rev = {}

def encode(col_value, map_rev):
    return [map_rev.get(col_value, col_value)]

def one_hot_encoding (df, cols_sparse):
  for col in cols_sparse:
    encoded = pd.get_dummies(df, dtype=float)
    return encoded
train_df_enc = one_hot_encoding(train_df, cols_sparse)
test_df_enc = one_hot_encoding(test_df, cols_sparse)
test_sample_enc = one_hot_encoding(test_sample, cols_sparse)

train_df_enc_np = train_df_enc.to_numpy()
test_df_enc_np = test_df_enc.to_numpy()
test_sample_enc_np = test_sample_enc.to_numpy()
train_df_enc_np

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()

    ground_truth = ground_truth[ground_truth.nonzero()].flatten()

    return sqrt(mean_squared_error(prediction, ground_truth))

"""Используем метод разложения, который называется **singular value decomposition** (SVD, cингулярное разложение). Смысл этого разложения в том, что исходную матрицу $X$ мы разбиваем на произведение ортогональных матриц $U$ и $V^T$ и диагональной матрицы $S$.

$$ X = UV^TS $$

В нашем случае $X$ – разреженная (состоящая преимущественно из нулей) user-item матрица. Разложив её на компоненты, мы можем их вновь перемножить и получить "восстановленную" матрицу $\hat{X}$. Матрица $\hat{X}$ и будет являться нашим предсказанием – метод SVD сделал сам за нас всё работу и заполнил пропуски в исходной матрице $X$

$$ UV^TS \approx \hat{X}$$
"""

import scipy.sparse as sp
from scipy.sparse.linalg import svds

# for i in range(len(train_df_enc_np[-1])):
for i in range(5):
  train_df_enc_np[-1][i] = 0.001

# делаем SVD
u, s, vt = svds(train_df_enc_np, k=10)

s_diag_matrix = np.diag(s)

s

s.shape

s_diag_matrix.shape

# предсказываем
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

X_pred.shape

# выводим метрику
print('Train user-based CF MSE: ' + str(rmse(X_pred, train_df_enc_np)))
print('Test user-based CF MSE: ' + str(rmse(X_pred, test_df_enc_np)))


"""# Predict function

Encode sample
"""

# len(cols_sparse)
train_df_enc = train_df.copy()


def get_columns_values_map (df, cols_sparse):
  columns_values_map = {}
  for col in cols_sparse:
    columns_values_map[col] = df[col].unique()
  return columns_values_map
columns_values_map = get_columns_values_map(train_df, cols_sparse)

def encode_by_column (df, columns_values_map):
  for col, col_value in columns_values_map.items():
    df[col]= pd.Categorical(df[col], categories = columns_values_map[col])
  return pd.get_dummies(df, dtype = float)

train_df_encoded_by_columns = encode_by_column(train_df, columns_values_map)

test_sample_enc = encode_by_column(test_sample, columns_values_map)

def get_feature_cols (all_columns, feature):
  cols = set()
  for col in all_columns:
    if len(col) >= len(feature):
      if col[:len(feature)] == feature and col[len(feature)] == '_':
        cols.add(col)
  return cols

def step_predict (df_enc_np, columns, columns_values_map, sample, steps, step_number):
  sample_enc = encode_by_column(sample, columns_values_map)
  df_enc_np = np.append(df_enc_np, encode_by_column(sample, columns_values_map), axis=0)

  # SVD
  u, s, vt = svds(df_enc_np, k=10)
  s_diag_matrix = np.diag(s)
  X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
  predicted_sample_df = pd.DataFrame([X_pred[-1]], columns=columns)

  predicts = {}
  # match step_number:
  #   case 1:
  for feature in steps[step_number]:
    cols = get_feature_cols(columns, feature)
    col_name_of_max = predicted_sample_df[list(cols)].idxmax(axis=1)
    predicts[feature] = col_name_of_max.iloc[0][len(feature)+1:]

  for key, value in predicts.items():
    if len(value) > 1 and (value[0] == '1' or value[0] == '0'):
      if float(value) == 0.:
        predicts[key] = False
      else:
        predicts[key] = True
    if len(value) == 1 and value.isdigit():
      predicts[key] = int(value)
  return predicts

train_df_encoded_by_columns_np = train_df_encoded_by_columns.to_numpy()

# for i in range(1, len(steps)):
#   print(i)
#   step_predict(train_df_enc_np, train_df_encoded_by_columns.columns, columns_values_map, test_sample, steps, i)

def get_default_sample (train_df_encoded_by_columns):
  sample = train_df_encoded_by_columns.iloc[0].copy()
  for col in train_df_encoded_by_columns.columns:
    if (sample[col]) == 1.:
      sample[col] = 0.
  return sample

from concurrent import futures
import logging

import grpc
import step_recomendation_pb2
import step_recomendation_pb2_grpc

sample = get_default_sample(train_df_encoded_by_columns)

def step0_to_df (sample, step0):
  sample["houseArea"] = step0.houseArea
  sample["siteArea"] = step0.siteArea
  sample["floorCount"] = step0.floorCount
  sample["region"] = step0.region
  sample["budgetFloor"] = step0.budgetFloor
  sample["budgetCeil"] = step0.budgetCeil

def step1_to_df (sample, step1):
  sample["siteChoosing"] = step1.SitePreparation.siteChoosing
  sample["geologicalWorks"] = step1.SitePreparation.geologicalWorks
  sample["geodeticalWorks"] = step1.SitePreparation.geodeticalWorks
  sample["cuttingBushesAndSmallForests"] = step1.SitePreparation.cuttingBushesAndSmallForests
  sample["clearingTheSiteOfDebris"] = step1.SitePreparation.clearingTheSiteOfDebris
  sample["cameras"] = step1.SiteWorks.cameras
  sample["temporaryFence"] = step1.SiteWorks.temporaryFence
  sample["homeProject"] = step1.HouseDesignAndProject.homeProject
  sample["designProject"] = step1.HouseDesignAndProject.designProject

def step2_to_df (sample, step2):
  sample["foundationType"] = step2.foundationType

def step3_to_df (sample, step3):
  sample["wallsMaterial"] = step3.wallsMaterial

def step4_to_df (sample, step4):
  sample["slopesNumber"] = step4.slopesNumber
  sample["roofType"] = step4.roofType

def step5_to_df (sample, step5):
  sample["facadeTechnology"] = step5.facadeTechnology

def step6_to_df (sample, step6):
  sample["windowMaterial"] = step6.windowMaterial
  sample["windowType"] = step6.windowType
  sample["doorMaterial"] = step6.doorMaterial
  
def electrician_to_df (sample, electrician):
  sample["plasticBoxesUpTo40mmWide"] = electrician.plasticBoxesUpTo40mmWide
  sample["layingAThreeToFive"] = electrician.layingAThreeToFive
  sample["cableLaying"] = electrician.cableLaying
  sample["installationOfTwoKey"] = electrician.installationOfTwoKey
  sample["installationOfSingleKey"] = electrician.installationOfSingleKey
  sample["recessedTypeSocketDevice"] = electrician.recessedTypeSocketDevice
  sample["installationOfPendant"] = electrician.installationOfPendant
  sample["chandeliersAndPendants"] = electrician.chandeliersAndPendants
  
def waterSupply_to_df (sample, water_supply):
  sample["layingOfInternalWaterSupplyPipelines"] = water_supply.layingOfInternalWaterSupplyPipelines
  sample["installationOfBathtubs"] = water_supply.installationOfBathtubs
  sample["installationOfSingle"] = water_supply.installationOfSingle
  sample["installationOfMixers"] = water_supply.installationOfMixers

def sewerage_to_df (sample, sewerage):
  sample["installationOfToilet"] = sewerage.installationOfToilet
  sample["layingOfSewerage50mm"] = sewerage.layingOfSewerage50mm
  sample["layingOfSewerage110mm"] = sewerage.layingOfSewerage110mm

def heating_to_df (sample, heating):
  sample["assemblyOfAWaterSupply"] = heating.assemblyOfAWaterSupply
  sample["layingOfInternalHeatingPipelines"] = heating.layingOfInternalHeatingPipelines
  sample["installationOfWindowFixtures"] = heating.installationOfWindowFixtures

def ventilation_to_df (sample, ventilation):
  sample["installationOfSplitSystems"] = ventilation.installationOfSplitSystems
  sample["cablingOnABrickWall"] = ventilation.cablingOnABrickWall

def step7_to_df (sample, step7):
  electrician_to_df(sample, step7.electrician)
  waterSupply_to_df(sample, step7.waterSupply)
  sewerage_to_df(sample, step7.sewerage)
  heating_to_df(sample, step7.heating)
  ventilation_to_df(sample, step7.ventilation)

def step8_to_df (sample, step8):
  sample["warmFloor"] = step8.warmFloor
  sample["ladderMaterial"] = step8.ladderMaterial

def step9_to_df (sample, step9):
  sample["wallDecoration"] = step9.wallDecoration
  sample["floorCovering"] = step9.floorCovering
  sample["ceilCovering"] = step9.ceilCovering

steps_to_df = [step0_to_df, step1_to_df, step2_to_df,
               step3_to_df, step4_to_df, step5_to_df,
               step6_to_df, step7_to_df, step8_to_df,
               step9_to_df]
def code_steps_to_df (sample, steps_info):
  for i in range(len(steps_info)):
    steps_to_df[i](sample, steps_info[i])


def sitePreparation_to_proto (predict):
  return step_recomendation_pb2.SitePreparation(
    siteChoosing = predict["siteChoosing"],
    geologicalWorks = predict["geologicalWorks"],
    geodeticalWorks = predict["geodeticalWorks"],
    cuttingBushesAndSmallForests = predict["cuttingBushesAndSmallForests"],
    clearingTheSiteOfDebris = predict["clearingTheSiteOfDebris"])

def siteWorks_to_proto (predict):
  return step_recomendation_pb2.SiteWorks(
    cameras = predict["cameras"],
    temporaryFence = predict["temporaryFence"]
  )

def houseDesignAndProject_to_proto (predict):
  return step_recomendation_pb2.HouseDesignAndProject(
    homeProject = predict["homeProject"],
    designProject = predict["designProject"]
  )

def step1_to_proto(predict):
  return step_recomendation_pb2.Step1Response(step1 = step_recomendation_pb2.Step1(sitePreparation = sitePreparation_to_proto(predict), 
                                                                    siteWorks = siteWorks_to_proto(predict),
                                                                    houseDesignAndProject = houseDesignAndProject_to_proto(predict)))



class Recomendation_system(step_recomendation_pb2_grpc.Recomendation_systemServicer):
    def recomend_step1(self, request, context):
        sample = get_default_sample(train_df_encoded_by_columns)
        
        steps_info = [request.step0]
        code_steps_to_df(sample, steps_info)

        predict = step_predict(train_df_encoded_by_columns_np, train_df_encoded_by_columns.columns, columns_values_map, test_sample, steps, 1)
        print(type(predict))
        print(predict)
        return step1_to_proto(predict)

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    step_recomendation_pb2_grpc.add_Recomendation_systemServicer_to_server(Recomendation_system(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()

