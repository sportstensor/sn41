import wandb
import schedule
from datetime import datetime, timedelta
import time

# Set your wandb project and entity
PROJECT_NAME = 'sportstensor-vali-logs'
ENTITY_NAME = 'sportstensor'
DAYS_TO_KEEP = 2  # Number of days to keep logs and artifacts

# Authenticate with wandb
wandb.login()

# Initialize the API
api = wandb.Api()

# Calculate the cutoff date
cutoff_date = datetime.now() - timedelta(days=DAYS_TO_KEEP)

# Function to delete old runs
def delete_old_runs():
    # Get all runs in the project
    runs = api.runs(f"{ENTITY_NAME}/{PROJECT_NAME}")

    for run in runs:
        # Convert the run's created_at timestamp to a datetime object
        run_created_at = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%SZ')

        # Check if the run is older than the cutoff date
        if run_created_at < cutoff_date:
            # Delete files
            files = run.files()
            for file in files:
                try:
                    print(f"Deleting file {file.name} in run {run.id}")
                    file.delete()
                except Exception as e:
                    print(f"Could not delete file {file.name}: {e}")

            # Delete artifacts with error handling
            for artifact in run.logged_artifacts():
                try:
                    print(f"Deleting artifact {artifact.id} in run {run.id}")
                    artifact.delete()
                except wandb.errors.CommError as e:
                    if "system managed artifact" in str(e):
                        print(f"Skipping system-managed artifact {artifact.id}")
                    else:
                        print(f"Could not delete artifact {artifact.id}: {e}")
                except Exception as e:
                    print(f"Could not delete artifact {artifact.id}: {e}")

            # Delete the run itself
            try:
                print(f"Deleting run {run.id} created at {run.created_at}")
                run.delete()
            except Exception as e:
                print(f"Could not delete run {run.id}: {e}")

    print("\n")
    print("Old logs and artifacts cleanup completed.")
    print("\n--------------------------------------------------------------------")

if __name__ == "__main__":
    # Schedule the function to run every hour
    schedule.every(60).minutes.do(delete_old_runs)

    # Delete old runs
    delete_old_runs()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(60)