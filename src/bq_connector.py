from google.cloud import bigquery
from google.cloud import bigquery_storage


class Singleton(type):

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class BigQueryClient(metaclass=Singleton):

    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        self.stclient = bigquery_storage.BigQueryReadClient()

    def dataset_exists(self, dataset_id):
        dataset_ref = self.client.dataset(dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
            return True
        except:
            return False

    def table_exists(self, dataset_id, table_id):
        table_ref = self.client.dataset(dataset_id).table(table_id)
        try:
            self.client.get_table(table_ref)
            return True
        except:
            return False

    def create_dataset(self, dataset_id):
        if not self.dataset_exists(dataset_id):
            print(f"Dataset '{dataset_id}' does not exist.")
            dataset_ref = self.client.dataset(dataset_id)
            dataset = bigquery.Dataset(dataset_ref)
            self.client.create_dataset(dataset)
            print(f"Dataset created: {dataset_id}")
        else:
            print(f"Dataset '{dataset_id}' exist.")

    def create_table(self, dataset_id, table_id, schema):
        if not self.table_exists(dataset_id, table_id):
            print(f"Table '{table_id}' does not exist.")
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table, exists_ok=True)
            print(f"Table created: {table_id}")
        else:
            print(f"Table '{table_id}' exist.")
