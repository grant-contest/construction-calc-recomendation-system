import numpy as np
import pandas as pd
import restrictions
import generate_df
import regions


"""# РСХБ"""

n_rows = 1000




regions = regions.get_regions()


train_test_quotient = 1
train_df = generate_df.gen_df(n_rows)
test_df = generate_df.gen_df(int(n_rows * train_test_quotient))
test_sample = generate_df.gen_df(1)

# val_df = generate_df(int(n_rows * 0.2 * 0.2))

steps_quantity = 11
steps = [None] * steps_quantity
steps[0] = {"houseArea", "siteArea", "floorCount", "region", "purpose", "budgetFloor", "budgetCeil"}
steps[1] = {"siteChoosing", "geologicalWorks", "geodeticalWorks", "cuttingBushesAndSmallForests", "clearingTheSiteOfDebris", "cameras", "temporaryFence", "homeProject", "designProject"}
steps[2] = {"foundationType"}
steps[3] = {"wallsMaterial"}
steps[4] = {"slopesNumber", "roofType"}
steps[5] = {"facadeTechnology"}
steps[6] = {"windowMaterial", "windowType", "doorMaterial"}
steps[7] = {"plasticBoxesUpTo40mmWide", "layingAThreeToFive", "cableLaying", "installationOfTwoKey", "installationOfSingleKey",
            "recessedTypeSocketDevice", "installationOfPendant", "chandeliersAndPendants", "layingOfInternalWaterSupplyPipelines",
            "installationOfBathtubs", "installationOfSingle", "installationOfMixers", "installationOfToilet", "layingOfSewerage50mm",
            "layingOfSewerage110mm", "assemblyOfAWaterSupply", "layingOfInternalHeatingPipelines", "installationOfWindowFixtures",
            "installationOfSplitSystems", "cablingOnABrickWall"}
steps[8] = {"warmFloor", "ladderMaterial"}
steps[9] = {"wallDecoration", "floorCovering", "ceilCovering"}
steps[10] = {"additionalBuildings"}

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
    "purpose",

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
    "installationOfMixers",

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
    "ceilCovering",
    
    # final
    "additionalBuildings",
    ]

assert len(train_df.columns) == len(cols_dense) + len(cols_sparse)

# len(train_df[cols_sparse])
train_df[cols_dense].head()

train_df[cols_sparse].dtypes

# train_df[cols_sparse]
train_df.dtypes

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
# print('Test user-based CF MSE: ' + str(rmse(X_pred, test_df_enc_np)))


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



def step_predict (df_enc_np, columns, columns_values_map, sample, steps, step_number, top):
  sample_enc = encode_by_column(sample, columns_values_map)
  df_enc_np = np.append(df_enc_np, encode_by_column(sample, columns_values_map), axis=0)

  # SVD
  u, s, vt = svds(df_enc_np, k=10)
  s_diag_matrix = np.diag(s)
  X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
  predicted_sample_df = pd.DataFrame([X_pred[-1]], columns=columns)

  predicts = {}
  # features = apply_restrictions(sample, steps[step_number].copy(), step_number)
  features = steps[step_number]
  for feature in features:
    cols = get_feature_cols(columns, feature)
    cols = restrictions.apply_restrictions(sample, cols, step_number, feature)
    if top == 1:
      col_name_of_max = predicted_sample_df[list(cols)].idxmax(axis=1)
      predicts[feature] = col_name_of_max.iloc[0][len(feature)+1:]
    if top == 3:
      tops = []
      for i in range(top):
        col_name_of_max = predicted_sample_df[list(cols)].idxmax(axis=1)
        tops.append(col_name_of_max.iloc[0][len(feature)+1:])
        cols.remove(col_name_of_max.iloc[0])
        # col_name_of_max = col_name_of_max.drop(columns = col_name_of_max.iloc[0])
        predicted_sample_df = predicted_sample_df.drop(columns = [col_name_of_max.iloc[0]])

      predicts[feature] = tops
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

def get_default_sample (train_df_encoded_by_columns):
  sample = train_df_encoded_by_columns.iloc[0].copy()
  for col in train_df_encoded_by_columns.columns:
    if (sample[col]) == 1.:
      sample[col] = 0.
  return sample

from concurrent import futures
import logging

import grpc
import sys
sys.path.append('proto/')

import proto.step_recomendation_pb2 as sr_pb2
import proto.step_recomendation_pb2_grpc as sr_pb2_grpc

sample = get_default_sample(train_df_encoded_by_columns)

def step0_to_df (sample, step0):
  sample["houseArea"] = step0.houseArea
  sample["siteArea"] = step0.siteArea
  sample["floorCount"] = step0.floorCount
  sample["region"] = step0.region
  sample["purpose"] = step0.purpose
  sample["budgetFloor"] = step0.budgetFloor
  sample["budgetCeil"] = step0.budgetCeil

def step1_to_df (sample, step1):
  sample["siteChoosing"] = step1.sitePreparation.siteChoosing
  sample["geologicalWorks"] = step1.sitePreparation.geologicalWorks
  sample["geodeticalWorks"] = step1.sitePreparation.geodeticalWorks
  sample["cuttingBushesAndSmallForests"] = step1.sitePreparation.cuttingBushesAndSmallForests
  sample["clearingTheSiteOfDebris"] = step1.sitePreparation.clearingTheSiteOfDebris
  sample["cameras"] = step1.siteWorks.cameras
  sample["temporaryFence"] = step1.siteWorks.temporaryFence
  sample["homeProject"] = step1.houseDesignAndProject.homeProject
  sample["designProject"] = step1.houseDesignAndProject.designProject

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

def step_final_to_df (sample, step_final):
  sample["additionalBuildings"] = step_final.additionalBuildings


steps_to_df = [step0_to_df, step1_to_df, step2_to_df,
               step3_to_df, step4_to_df, step5_to_df,
               step6_to_df, step7_to_df, step8_to_df,
               step9_to_df, step_final_to_df]

def code_steps_to_df (sample, steps_info):
  for i in range(len(steps_info)):
    steps_to_df[i](sample, steps_info[i])

def sitePreparation_to_proto (predict):
  return sr_pb2.SitePreparation(
    siteChoosing = predict["siteChoosing"],
    geologicalWorks = predict["geologicalWorks"],
    geodeticalWorks = predict["geodeticalWorks"],
    cuttingBushesAndSmallForests = predict["cuttingBushesAndSmallForests"],
    clearingTheSiteOfDebris = predict["clearingTheSiteOfDebris"])

def siteWorks_to_proto (predict):
  return sr_pb2.SiteWorks(
    cameras = predict["cameras"],
    temporaryFence = predict["temporaryFence"]
  )

def houseDesignAndProject_to_proto (predict):
  return sr_pb2.HouseDesignAndProject(
    homeProject = predict["homeProject"],
    designProject = predict["designProject"]
  )

def step1_to_proto(predict):
  return sr_pb2.Step1Response(step1 = sr_pb2.Step1(sitePreparation = sitePreparation_to_proto(predict), 
                                                                    siteWorks = siteWorks_to_proto(predict),
                                                                    houseDesignAndProject = houseDesignAndProject_to_proto(predict)))

def step2_to_proto(predict):
  return sr_pb2.Step2Response(step2 = sr_pb2.Step2(foundationType = predict["foundationType"]))

def step3_to_proto(predict):
  return sr_pb2.Step3Response(step3 = sr_pb2.Step3(wallsMaterial = predict["wallsMaterial"]))

def step4_to_proto(predict):
  return sr_pb2.Step4Response(step4 = sr_pb2.Step4(slopesNumber = predict["slopesNumber"], roofType = predict["roofType"]))

def step5_to_proto(predict):
  return sr_pb2.Step5Response(step5 = sr_pb2.Step5(facadeTechnology = predict["facadeTechnology"]))

def step6_to_proto(predict):
  return sr_pb2.Step6Response(step6 = sr_pb2.Step6(windowMaterial = predict["windowMaterial"], 
                                                   windowType = predict["windowType"],
                                                   doorMaterial = predict["doorMaterial"]))

def electrician_to_proto (predict):
  return sr_pb2.Electrician(
    plasticBoxesUpTo40mmWide = predict["plasticBoxesUpTo40mmWide"],
    layingAThreeToFive = predict["layingAThreeToFive"],
    cableLaying = predict["cableLaying"],
    installationOfTwoKey = predict["installationOfTwoKey"],
    installationOfSingleKey = predict["installationOfSingleKey"],
    recessedTypeSocketDevice = predict["recessedTypeSocketDevice"],
    installationOfPendant = predict["installationOfPendant"],
    chandeliersAndPendants = predict["chandeliersAndPendants"]
  )

def waterSupply_to_proto (predict):
  return sr_pb2.WaterSupply(
    layingOfInternalWaterSupplyPipelines = predict["layingOfInternalWaterSupplyPipelines"],
    installationOfBathtubs = predict["installationOfBathtubs"],
    installationOfSingle = predict["installationOfSingle"],
    installationOfMixers = predict["installationOfMixers"]
  )

def sewerage_to_proto (predict):
  return sr_pb2.Sewerage(
    installationOfToilet = predict["installationOfToilet"],
    layingOfSewerage50mm = predict["layingOfSewerage50mm"],
    layingOfSewerage110mm = predict["layingOfSewerage110mm"]
  )

def heating_to_proto (predict):
  return sr_pb2.Heating(
    assemblyOfAWaterSupply = predict["assemblyOfAWaterSupply"],
    layingOfInternalHeatingPipelines = predict["layingOfInternalHeatingPipelines"],
    installationOfWindowFixtures = predict["installationOfWindowFixtures"]
  )

def ventilation_to_proto (predict):
  return sr_pb2.Ventilation(
    installationOfSplitSystems = predict["installationOfSplitSystems"],
    cablingOnABrickWall = predict["cablingOnABrickWall"]
  )

def step7_to_proto(predict):
  return sr_pb2.Step7Response(step7 = sr_pb2.Step7(electrician = electrician_to_proto(predict), 
                                                   waterSupply = waterSupply_to_proto(predict),
                                                   sewerage = sewerage_to_proto(predict),
                                                   heating = heating_to_proto(predict),
                                                   ventilation = ventilation_to_proto(predict)
                                                   ))

def step8_to_proto(predict):
  return sr_pb2.Step8Response(step8 = sr_pb2.Step8(warmFloor = predict["warmFloor"], 
                                                   ladderMaterial = predict["ladderMaterial"]
                                                   ))

def step9_to_proto(predict):
  return sr_pb2.Step9Response(step9 = sr_pb2.Step9(wallDecoration = predict["wallDecoration"], 
                                                   floorCovering = predict["floorCovering"],
                                                   ceilCovering = predict["ceilCovering"]
                                                   ))

def step_final_to_proto(predict):
  return sr_pb2.FinalResponse(stepFinal = sr_pb2.StepFinal(additionalBuildings1 = predict["additionalBuildings"][0],
                                                           additionalBuildings2 = predict["additionalBuildings"][1],
                                                           additionalBuildings3 = predict["additionalBuildings"][2],
                                                   ))


def get_steps_info(request, step_number):
  steps_info = []
  for i in range(step_number):
    match i:
      case 0:
        steps_info.append(request.step0) 
      case 1:
        steps_info.append(request.step1) 
      case 2:
        steps_info.append(request.step2) 
      case 3:
        steps_info.append(request.step3) 
      case 4:
        steps_info.append(request.step4) 
      case 5:
        steps_info.append(request.step5) 
      case 6:
        steps_info.append(request.step6) 
      case 7:
        steps_info.append(request.step7) 
      case 8:
        steps_info.append(request.step8) 
      case 9:
        steps_info.append(request.step9) 
  return steps_info

stepx_to_proto = [step1_to_proto, step2_to_proto, step3_to_proto, step4_to_proto, step5_to_proto, 
                  step6_to_proto, step7_to_proto, step8_to_proto, step9_to_proto, step_final_to_proto]

def recomend_stepx(self, request, context, x):
    sample = get_default_sample(train_df_encoded_by_columns)
    steps_info = get_steps_info(request, x)
    code_steps_to_df(sample, steps_info)
    top = 1
    if x == 10:
      top = 3
    predict = step_predict(train_df_encoded_by_columns_np, train_df_encoded_by_columns.columns, columns_values_map, test_sample, steps, x, top)
    print(predict)
    # if top == 1:
    return stepx_to_proto[x-1](predict)

class Recomendation_system(sr_pb2_grpc.Recomendation_systemServicer):    
    def recomend_step1(self, request, context):
      return recomend_stepx(self, request, context, 1)
    def recomend_step2(self, request, context):
      return recomend_stepx(self, request, context, 2)
    def recomend_step3(self, request, context):
      return recomend_stepx(self, request, context, 3)
    def recomend_step4(self, request, context):
      return recomend_stepx(self, request, context, 4)
    def recomend_step5(self, request, context):
      return recomend_stepx(self, request, context, 5)
    def recomend_step6(self, request, context):
      return recomend_stepx(self, request, context, 6)
    def recomend_step7(self, request, context):
      return recomend_stepx(self, request, context, 7)
    def recomend_step8(self, request, context):
      return recomend_stepx(self, request, context, 8)
    def recomend_step9(self, request, context):
      return recomend_stepx(self, request, context, 9)
    def recomend_final(self, request, context):
      return recomend_stepx(self, request, context, 10)

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sr_pb2_grpc.add_Recomendation_systemServicer_to_server(Recomendation_system(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()


