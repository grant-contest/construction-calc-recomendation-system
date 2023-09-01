import numpy as np

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
  return np.random.choice(["Дерево", "Бетон", "Металл"])

def foundationTypeRestriction(floorCount):
  if floorCount == 3:
    return np.random.choice(["Ленточный", "Плитный"])
  return np.random.choice(["Свайный", "Столбчатый", "Ленточный", "Плитный"])


def apply_restrictions(sample, cols, step_number, feature):
  cols_copy = cols.copy()
  print("Before restrictions: ", cols)
  for col in cols_copy:
    cols.remove(col)
    cols.add(col[len(feature)+1:])
  if step_number == 3 and sample["foundationType"].item() in {"Свайный", "Столбчатый"}:
    cols = {"Дерево","Каркас"}
  if step_number == 5:
    match sample["wallsMaterial"].item():
      case "Каркас":
        cols = {"Панели (сайдинг)", "Облицовка кирпичом", "Искусственный камень"}
      case "Дерево":
        cols = {"Без отделки", "Панели (сайдинг)"}
      case "Кирпич":
        cols = {"Без отделки"}
  if step_number == 8 and feature == "ladderMaterial":
    if sample["floorCount"].item() == 1:
      cols = {"-"}
    if sample["foundationType"].item() in {"Свайный", "Столбчатый"}:
      cols = {"Дерево", "Металл"}
  if step_number == 2 and (sample["floorCount"].item() == 3):
    cols = {"Ленточный", "Плитный"}
  cols_copy = cols.copy()
  for col in cols_copy:
    cols.remove(col)
    cols.add(feature + '_' + col)
  print("After restrictions: ", cols)
  return cols
