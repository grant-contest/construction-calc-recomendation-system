import random
import regions
import restrictions
import pandas as pd
import numpy as np
import additional_buildings

def any_to_0_1(a):
  if a > 0:
    return 1.
  else:
    return 0.
    
def gen_df(n_rows):

  floorCounts = [(float(random.randint(1, 3))) for i in range(n_rows)]
  foundationTypes = [restrictions.foundationTypeRestriction(floorCounts[i]) for i in range(n_rows)]
  wallsMaterial = [restrictions.wallsMaterialRestriction(foundationTypes[i]) for i in range(n_rows)]

  add_buildings_db = additional_buildings.init_additional_buildings_db()
  add_buildings_names = []
  for key, value in add_buildings_db.items():
    add_buildings_names.append(key)

  regs = regions.get_regions() 
  df = pd.DataFrame({
  # step 0
  "houseArea": [float(random.randint(25, 250)) for i in range(n_rows)],
  "siteArea": [float(random.randint(0, 100)) for i in range(n_rows)],
  "floorCount": floorCounts,
  "region": [np.random.choice(regs) for i in range(n_rows)],
  "purpose": [np.random.choice(["Постоянное место жительства", "Место отдыха, 'Дача'", "Место работы"]) for i in range(n_rows)],
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
  "facadeTechnology": [restrictions.facadeTechnologyRestriction(wallsMaterial[i]) for i in range(n_rows)],

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
  "installationOfMixers": [any_to_0_1(np.random.randn()) for i in range(n_rows)],

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

  # step 8
  "warmFloor": [any_to_0_1(np.random.randn()) for i in range(n_rows)],
  "ladderMaterial": [restrictions.ladderMaterialRestriction(floorCounts[i], foundationTypes[i]) for i in range(n_rows)],

  # step 9
  "wallDecoration": [np.random.choice(["Декоративная штукатурка", "Покраска", "Обои", "Плитка"]) for i in range(n_rows)],
  "floorCovering": [np.random.choice(["Ламинат", "Линолеум"]) for i in range(n_rows)],
  "ceilCovering": [np.random.choice(["Натяжной потолок", "Окраска", "Обои", "Штукатурка"]) for i in range(n_rows)],

  # final
  "additionalBuildings": [np.random.choice(add_buildings_names) for i in range(n_rows)],

  })

  return df
