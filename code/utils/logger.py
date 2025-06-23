import logging
import numpy as np


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
def result_log(Indicators="Accuracy",target="",source="",results=None):
    results = [round(x, 2) for x in results]
    mean = np.mean(results)
    std = np.std(results)
    results.append(f"{mean:.3f}±{std:.3f}")
    logging.info(f"Indicators:{Indicators} Task:{target}-{source} Mean_Var: {results}")



def result_log_all(Indicators="Accuracy",result_1=None,result_2=None):
    mean_acc = np.mean(result_1)
    mean_std = np.mean(result_2)
    result_1.append(mean_acc)
    result_2.append(mean_std)
    formatted_values = [f"{mean:.3f}±{error:.3f}" for mean, error in zip(result_1, result_2)]
    logging.info(f'Indicators: {Indicators}\nMean_Var: {formatted_values}')

