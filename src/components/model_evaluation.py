
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")

        df["Gender"] = df["Gender"].map({
            "Female": 0,
            "Male": 1
        }).astype(int)

        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")

        df = pd.get_dummies(df, drop_first=True)

        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""

        logging.info("Renaming specific columns and casting to int")

        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })

        for col in [
            "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes"
        ]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def _drop_id_column(self, df):
        """Drop the '_id' column if it exists."""

        logging.info("Dropping '_id' column")

        if "_id" in df.columns:
            df = df.drop("_id", axis=1)

        return df

    def evaluate_model(self) -> EvaluateModelResponse:

        try:

            logging.info("Starting local model evaluation.")

            # Load test data
            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info(
                f"Test data loaded. Shape: {test_df.shape}"
            )

            # Separate input and target
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            logging.info(
                "Input and target columns separated."
            )

            # Apply same transformations used during training
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            logging.info(
                "Custom transformations applied to test data."
            )

            # Load trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            logging.info(
                "Trained model loaded successfully."
            )

            # Make predictions
            y_pred = trained_model.predict(x)

            logging.info(
                "Predictions generated successfully."
            )

            # Calculate F1 score
            trained_model_f1_score = f1_score(
                y,
                y_pred
            )

            logging.info(
                f"F1 Score of trained model: {trained_model_f1_score}"
            )

            # Since we don't have a production/AWS model yet,
            # there is no old model to compare against.
            best_model_f1_score = None

            # Accept the current model because it is the first
            # model being evaluated locally.
            is_model_accepted = True

            difference = trained_model_f1_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

            logging.info(
                f"Model evaluation result: {result}"
            )

            return result

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:

            print(
                "------------------------------------------------------------------------------------------------"
            )

            logging.info(
                "Initialized Model Evaluation Component."
            )

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=None,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(
                f"Model evaluation artifact: {model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
