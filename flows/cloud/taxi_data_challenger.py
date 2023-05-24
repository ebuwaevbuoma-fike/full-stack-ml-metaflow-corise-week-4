from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger
from metaflow import project, S3
from metaflow.cards import Markdown, Table, Image, Artifact

# URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
URL = 's3://outerbounds-datasets/taxi/latest.parquet'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2', 'xgboost' : '1.7.4'})
@project(name='taxifare_prediction')
class TaxiFarePrediction(FlowSpec):

    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        # TODO: 
        # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
        # Understand what is happening.
        # Revisit task 1 and think about what might go in this function.

        obviously_bad_data_filters = [
            df.fare_amount > 0,         # fare_amount in US Dollars
            df.trip_distance <= 100,    # trip_distance in miles
            df.trip_distance > 0,
            df.passenger_count <= 5 #most taxis can only sit 4 passengers
        ]
        for f in obviously_bad_data_filters:
            df = df[f]

        return df

    @step
    def start(self):

        import pandas as pd
        from sklearn.model_selection import train_test_split

        with S3() as s3:
            obj = s3.get(URL)
            df = pd.read_parquet(obj.path)

        self.df = self.transform_features(df)
        # self.df = self.transform_features(pd.read_parquet(self.data_url))

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
      
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.xgboost_model)
    
    @step
    def xgboost_model(self):
        "Fit an XGBoost regressor"
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.next(self.validate)

    def gather_sibling_flow_run_results(self):

        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow 
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅" 
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [Markdown(icon), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(score), Markdown(msg)]
                rows.append(row)
            else:
                rows.append([Markdown("✅"), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(str(self.scores.mean())), Markdown("This run...")])
        return rows

    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(Table(self.gather_sibling_flow_run_results(), headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"]))
        self.next(self.end)

    @step
    def end(self):
        self.model_name = 'challenger'
        self.model_type = "xgboost"
        print(f'{self.model_name} score: {self.scores.mean():.2f}')
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
