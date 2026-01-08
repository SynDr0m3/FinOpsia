from pathlib import Path
import pandas as pd
from monitoring.logger import logger

def read_transactions(file_path: str) -> pd.DataFrame:
  """
  Read transaction data from a CSV file.

  Args:
    file_path (str): Path to the CSV file.

  Returns:
    pd.DataFrame: Raw transaction data.

  Raises:
    FileNotFoundError: If file does not exist.
    ValueError: If file is empty.
 
  """
  path = Path(file_path)

  logger.info(f"Attempting to read transactions from {path}")

  if not path.exists():
    logger.error(f"File not found: {path}")
    raise FileNotFoundError(f"Transaction not found: {path}")
  
  df = pd.read_csv(path)

  if df.empty:
    logger.error("Transaction file is empty")
    raise ValueError("Transaction file is empty")
  
  logger.info(
    f"Loaded {len(df)} transactions"
    f"with columns: {list(df.columns)}"
  )

  return df 

  

