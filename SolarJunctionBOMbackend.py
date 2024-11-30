import os
import pandas as pd
import math
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.memory import ConversationBufferMemory
import traceback
from typing import List, Dict
from io import BytesIO
from urllib.parse import quote_plus
import streamlit as st
from datetime import datetime
import ast
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def init_database():
    try:
        db_config = {
            'server': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', '3306'),  
            'database': os.getenv('DB_NAME'),
            'username': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
        }

        missing_credentials = [key for key, value in db_config.items() if not value]
        if missing_credentials:
            raise ValueError(f"Missing required database credentials: {', '.join(missing_credentials)}")

        try:
            db_config['port'] = db_config['port'].strip().rstrip(';')
            port = int(db_config['port'])
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
        except ValueError as e:
            raise ValueError(f"Invalid database port: {db_config['port']}. Error: {str(e)}")

        connection_url = (
            f"mysql+pymysql://{db_config['username']}:{quote_plus(db_config['password'])}"
            f"@{db_config['server']}:{db_config['port']}/{db_config['database']}"
        )

        engine = create_engine(connection_url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

        db = SQLDatabase.from_uri(connection_url, view_support=True)

        return engine, db

    except Exception as e:
        error_msg = f"Database initialization failed: {str(e)}"
        print(error_msg)
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n\n")
        raise Exception(error_msg)

@st.cache_resource
def init_llm():
    """Initialize LLM (cached)"""
    return ChatOpenAI(temperature=0, model="gpt-4o-mini")

@st.cache_data
def get_module_data(_engine, manufacturer: str = None):
    """Get module data from MySQL database (cached)"""
    query = "SELECT * FROM solar_products"
    if manufacturer:
        query += f" WHERE Make = '{manufacturer}'"
    return pd.read_sql(query, _engine)

@st.cache_data
def get_inverter_data(_engine):
    """Get inverter data from MySQL database (cached)"""
    return pd.read_sql("SELECT * FROM inverter_data", _engine)

@st.cache_data
def get_available_earthing_material_manufacturers(_engine) -> List[str]:
    """Get list of available earthing material and lightning arrestor manufacturers (cached)"""
    query = "SELECT DISTINCT Manufacturer FROM Earthing_Material_And_Lightning_Arrestor ORDER BY Manufacturer"
    result = pd.read_sql(query, _engine)
    return result['Manufacturer'].tolist()

def get_available_earthing_rod_manufacturers(_engine) -> List[str]:
    """Get list of available earthing rod manufacturers (cached)"""
    query = "SELECT DISTINCT Manufacturer FROM Earthing_rod ORDER BY SKU"
    result = pd.read_sql(query, _engine)
    return result['SKU'].tolist()


@st.cache_data
def get_available_manufacturers(_engine) -> List[str]:
    """Get list of available manufacturers (cached)"""
    query = "SELECT DISTINCT Make FROM solar_products ORDER BY Make"
    result = pd.read_sql(query, _engine)
    return result['Make'].tolist()

@st.cache_data
def get_available_inverters(_engine) -> List[str]:
    """Get list of available inverter manufacturers (cached)"""
    query = "SELECT DISTINCT Manufacturer FROM inverter_data ORDER BY Manufacturer"
    result = pd.read_sql(query, _engine)
    return result['Manufacturer'].tolist()

class SolarBOQGenerator:
    def __init__(self):
        self.engine, self.db = init_database()
        self.llm = init_llm()
        # Initialize toolkit and agent
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=False,
            handle_parsing_errors=True, 
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

    def log_error(self, error):
        """Log errors to file"""
        with open("error_log.txt", "a") as f:
            f.write(f"Error: {str(error)}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n\n")

    def calculate_pv_modules(self, capacity: float, module_wattage: float) -> int:
        """Calculate number of PV modules needed"""
        return math.ceil(capacity * 1000 / module_wattage)

    def calculate_strings_and_mc4(self, num_modules: int, vmp: float) -> tuple:
        """Calculate number of strings and MC4 connectors needed"""
        modules_per_string = int(1000 / vmp)
        num_strings = num_modules // modules_per_string + (1 if num_modules % modules_per_string else 0)
        mc4_connectors = num_strings * 2
        return num_strings, mc4_connectors

    def select_inverter_rating(self, capacity: float, inverter_manufacturer: str = None) -> float:
        """
        Select inverter rating closest to requested capacity for the specified manufacturer.
        
        Args:
            capacity (float): Project capacity in KW
            inverter_manufacturer (str, optional): Selected inverter manufacturer
            
        Returns:
            float: Selected inverter rating in KW
        """
        try:
            # Convert project capacity from KW to W
            target_capacity = int(capacity * 1000)
            
            # Print debug information (commented out)
            # print(f"Searching for inverter with target capacity: {target_capacity}W")
            # if inverter_manufacturer:
            #     print(f"Manufacturer filter: {inverter_manufacturer}")
            
            # Base query to select the inverter with the closest capacity
            query = """
                SELECT Capacity, Manufacturer
                FROM inverter_data
                WHERE Capacity > 0
            """
            
            # Add manufacturer filter if specified and not "Not specified"
            if inverter_manufacturer and inverter_manufacturer.strip() and inverter_manufacturer != "Not specified":
                query += f" AND Manufacturer = '{inverter_manufacturer}'"
                
            # Add ordering by absolute difference and limit
            query += f"""
                ORDER BY ABS(CAST(REPLACE(REPLACE(Capacity, 'W', ''), ' ', '') AS SIGNED) - {target_capacity})
                LIMIT 1
            """
            
            # Execute query and get result
            result = pd.read_sql(query, self.engine)
            
            # Debug print the result (commented out)
            # print(f"Query result: {result.to_dict('records')}")
            
            if result.empty:
                # print("No inverter found matching criteria")  # Commented out
                return 0
            
            try:
                # Clean the capacity string by removing 'W' and spaces, then convert to float
                capacity_str = str(result['Capacity'].iloc[0])
                capacity_value = float(capacity_str.replace('W', '').replace(' ', ''))
                
                # Convert back to KW
                kw_rating = capacity_value / 1000
                
                # Print selected rating (commented out)
                # print(f"Selected inverter rating: {kw_rating}kW")
                return kw_rating
                
            except (ValueError, TypeError) as e:
                # Error handling (commented out)
                # print(f"Error converting capacity value: {str(e)}")
                # print(f"Raw capacity value: {result['Capacity'].iloc[0]}")
                return 0
        except Exception as e:
            self.log_error(e)
            # Error handling (commented out)
            # print(f"Error in select_inverter_rating: {str(e)}")
            # traceback.print_exc()
            return 0

    def select_inverter_and_rate(self, capacity: float, inverter_manufacturer: str = None) -> tuple:
        """
        Select inverter rating and rate closest to requested capacity for the specified manufacturer.
        
        Args:
            capacity (float): Project capacity in KW
            inverter_manufacturer (str, optional): Selected inverter manufacturer
            
        Returns:
            tuple: (selected_rating_kw, rate)
                - selected_rating_kw (float): Selected inverter rating in KW
                - rate (float): Rate for the selected inverter
        """
        try:
            # Convert project capacity from KW to W
            target_capacity = int(capacity * 1000)
            
            # Print debug information (commented out)
            # print(f"Searching for inverter with target capacity: {target_capacity}W")
            # if inverter_manufacturer:
            #     print(f"Manufacturer filter: {inverter_manufacturer}")
            
            # Base query to select the inverter with the closest capacity and its rate
            query = """
                SELECT Capacity, Manufacturer, Rate
                FROM inverter_data
                WHERE Capacity > 0
            """
            
            # Add manufacturer filter if specified and not "Not specified"
            if inverter_manufacturer and inverter_manufacturer.strip() and inverter_manufacturer != "Not specified":
                query += f" AND Manufacturer = '{inverter_manufacturer}'"
                
            # Add ordering by absolute difference and limit
            query += f"""
                ORDER BY ABS(CAST(REPLACE(REPLACE(Capacity, 'W', ''), ' ', '') AS SIGNED) - {target_capacity})
                LIMIT 1
            """
            
            # Execute query and get result
            result = pd.read_sql(query, self.engine)
            
            # Debug print the result (commented out)
            # print(f"Query result: {result.to_dict('records')}")
            
            if result.empty:
                # print("No inverter found matching criteria")  # Commented out
                return 0, 0
            
            try:
                # Clean the capacity string and convert to float
                capacity_str = str(result['Capacity'].iloc[0])
                capacity_value = float(capacity_str.replace('W', '').replace(' ', ''))
                
                # Get the rate
                rate = float(result['Rate'].iloc[0]) if pd.notnull(result['Rate'].iloc[0]) else 0
                
                # Convert capacity to KW
                kw_rating = capacity_value / 1000
                
                # Print selected rating and rate (commented out)
                # print(f"Selected inverter rating: {kw_rating}kW, Rate: {rate}")
                return kw_rating, rate
                
            except (ValueError, TypeError) as e:
                # Error handling (commented out)
                # print(f"Error converting values: {str(e)}")
                # print(f"Raw capacity value: {result['Capacity'].iloc[0]}")
                # print(f"Raw rate value: {result['Rate'].iloc[0]}")
                return 0, 0
                
        except Exception as e:
            self.log_error(e)
            # Error handling (commented out)
            # print(f"Error in select_inverter_and_rate: {str(e)}")
            # traceback.print_exc()
            return 0, 0
        
        
    def get_earthing_rod_manufacturers(self) -> List[str]:
        """Get list of available earthing rod manufacturers"""
        query = "SELECT DISTINCT SKU FROM Earthing_rod ORDER BY SKU"
        result = pd.read_sql(query, self.engine)
        return result['SKU'].tolist()

    def get_earthing_rod_rate(self, manufacturer: str) -> float:
        """Get rate for earthing rod from selected manufacturer"""
        try:
            query = f"""
                SELECT Rate 
                FROM Earthing_rod 
                WHERE SKU = '{manufacturer}'
                LIMIT 1
            """
            result = pd.read_sql(query, self.engine)
            return float(result['Rate'].iloc[0]) if not result.empty else 0
        except Exception as e:
            self.log_error(e)
            return 0
        

    def generate_boq(self, project_capacity: float, selected_module: Dict,
                manufacturer: str, inverter_manufacturer: str = None,
                earthing_material_manufacturer: str = None,
                earthing_rod_manufacturer: str = None) -> List[Dict]:
        """Generate Bill of Quantities with formatted currency values and proper type handling"""
        try:
            # Ensure selected_module contains numeric values for Pmax, Vmp, and Rate
            pmax_w = float(selected_module['Pmax'])
            vmp_v = float(selected_module['Vmp'])
            pv_module_rate = float(selected_module['Rate'])

            # Get inverter rating and rate using the new method
            inverter_rating, inverter_rate = self.select_inverter_and_rate(project_capacity, inverter_manufacturer)

            num_pv_modules = self.calculate_pv_modules(project_capacity, pmax_w)
            num_strings, mc4_connectors = self.calculate_strings_and_mc4(num_pv_modules, vmp_v)
            earthing_material_rate = float(self.get_earthing_material_rate(earthing_material_manufacturer)) if earthing_material_manufacturer else 0
            earthing_rod_rate = float(self.get_earthing_rod_rate(earthing_rod_manufacturer)) if earthing_rod_manufacturer else 0

            # Helper function to ensure numeric values
            def ensure_numeric(value):
                if value == '' or value is None:
                    return 0.0
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            # Helper function for currency formatting
            def format_currency(amount: float | int) -> str:
                if amount == 0:
                    return "-"
                amount = float(amount)
                if amount == int(amount):
                    return f"{int(amount):,}"
                return f"{amount:,.2f}"

            # List of all BOQ items with explicit type handling
            boq_data = [
                {
                    "Item Name": "PV Modules",
                    "Quantity": str(num_pv_modules),  # Convert to string to avoid serialization issues
                    "Unit(s)": "Nos",
                    "Rating": f"{pmax_w}W",
                    "Make": manufacturer,
                    "Rate": format_currency(pv_module_rate)
                },
                {
                    "Item Name": "Inverters",
                    "Quantity": "1",  # Use string instead of int
                    "Unit(s)": "Nos",
                    "Rating": f"{inverter_rating}kW" if inverter_rating else "-",
                    "Make": inverter_manufacturer if inverter_manufacturer else "-",
                    "Rate": format_currency(inverter_rate)
                },
                {
                    "Item Name": "MC4 Connectors",
                    "Quantity": str(mc4_connectors),  # Convert to string
                    "Unit(s)": "Pairs",
                    "Rating": "-",
                    "Make": "-",
                    "Rate": format_currency(0)
                },
                {
                    "Item Name": "Earthing Rod",
                    "Quantity": "3",  # Use string instead of int
                    "Unit(s)": "Nos",
                    "Rating": "-",
                    "Make": earthing_rod_manufacturer if earthing_rod_manufacturer else "-",
                    "Rate": format_currency(earthing_rod_rate)
                },
                {
                    "Item Name": "Remote Monitoring",
                    "Quantity": "1",  # Use string instead of int
                    "Unit(s)": "Nos",
                    "Rating": "-",
                    "Make": "-",
                    "Rate": format_currency(0)
                },
                {
                    "Item Name": "Earthing Material and Lightning Arrestor",
                    "Quantity": "3",  # Use string instead of int
                    "Unit(s)": "Nos",
                    "Rating": "-",
                    "Make": earthing_material_manufacturer if earthing_material_manufacturer else "-",
                    "Rate": format_currency(earthing_material_rate)
                },
                {
                    "Item Name": "Danger Board & Signage",
                    "Quantity": "1",  # Use string instead of int
                    "Unit(s)": "Nos",
                    "Rating": "-",
                    "Make": "-",
                    "Rate": format_currency(0)
                }
            ]

            # Calculate costs for each item
            for item in boq_data:
                rate = ensure_numeric(item["Rate"].replace(",", "")) if item["Rate"] != "-" else 0
                quantity = float(item["Quantity"])  # Convert string to float for calculation
                cost = quantity * rate
                item["Cost"] = format_currency(cost)

            # Calculate total cost
            total_cost = sum(
                ensure_numeric(item["Cost"].replace(",", "")) if item["Cost"] != "-" else 0
                for item in boq_data
            )

            # Add the total sum row with empty strings instead of None
            boq_data.append({
                "Item Name": "Total",
                "Quantity": "",  # Empty string instead of None
                "Unit(s)": "",   # Empty string instead of None
                "Rating": "",    # Empty string instead of None
                "Make": "",      # Empty string instead of None
                "Rate": "",      # Empty string instead of None
                "Cost": format_currency(total_cost)
            })

            return boq_data
        except Exception as e:
            print(f"Error in generate_boq: {str(e)}")
            traceback.print_exc()
            return []

    def generate_csv(self, boq_data: List[Dict]) -> bytes:
        """Generate CSV file from BOQ data"""
        output = BytesIO()
        pd.DataFrame(boq_data).to_csv(output, index=False)
        return output.getvalue()
    
    def get_available_earthing_material_manufacturers(self) -> List[str]:
        """Get list of available earthing material manufacturers"""
        return get_available_earthing_material_manufacturers(self.engine)

    def get_earthing_material_data(self, manufacturer: str = None):
        """Get earthing material data for a specific manufacturer or all manufacturers"""
        query = "SELECT * FROM Earthing_Material_And_Lightning_Arrestor"
        if manufacturer:
            query += f" WHERE Manufacturer = '{manufacturer}'"
        return pd.read_sql(query, self.engine)
    
    def get_earthing_material_rate(self, manufacturer: str) -> float:
        """Get rate for earthing material from selected manufacturer"""
        try:
            query = f"""
                SELECT Rate 
                FROM Earthing_Material_And_Lightning_Arrestor 
                WHERE Manufacturer = '{manufacturer}'
                LIMIT 1
            """
            result = pd.read_sql(query, self.engine)
            return float(result['Rate'].iloc[0]) if not result.empty else 0
        except Exception as e:
            self.log_error(e)
            return 0
        
    def get_available_earthing_rod_manufacturers(self) -> List[str]:
        """Get list of available earthing rod manufacturers"""
        return get_available_earthing_rod_manufacturers(self.engine)

    def get_earthing_rod_data(self, manufacturer: str = None):
        """Get earthing rod data for a specific manufacturer or all manufacturers"""
        query = "SELECT * FROM Earthing_rod"
        if manufacturer:
            query += f" WHERE SKU = '{manufacturer}'"
        return pd.read_sql(query, self.engine)
    
    def get_earthing_rod_rate(self, manufacturer: str) -> float:
        """Get rate for earthing rod from selected manufacturer"""
        try:
            query = f"""
                SELECT Rate 
                FROM Earthing_rod 
                WHERE SKU = '{manufacturer}'
                LIMIT 1
            """
            result = pd.read_sql(query, self.engine)
            return float(result['Rate'].iloc[0]) if not result.empty else 0
        except Exception as e:
            self.log_error(e)
            return 0
        
    
   
    def extract_info_from_input(self, user_input: str) -> Dict:
        """Extract project details from user input"""
        query = f"""
        System Prefix:
        You are an agent designed to interact with a MYSQL database.
        Given an input question, create a syntactically correct dialect query to run, then look at the results of the query and return the answer. 
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most relevent results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        Description of the tables from the database:
        CREATE descibe solar_products (
            product_id BIGINT PRIMARY KEY,
            SKU VARCHAR(50),
            Make VARCHAR(100),
            Model VARCHAR(255),
            Pmax DECIMAL(5,2),      -- maximum power
            Vmp DECIMAL(5,2),       -- maximum voltage
            Imp DECIMAL(5,2),       -- maximum current
            Isc DECIMAL(5,2),       -- short-circuit current
            Voc DECIMAL(5,2),       -- open-circuit voltage
            Weight DECIMAL(6,2),    -- in kg or grams
            Height DECIMAL(6,2),    -- in mm or cm
            Depth DECIMAL(6,2),     -- in mm or cm
            Width DECIMAL(6,2),     -- in mm or cm
            Rate DECIMAL(10,2)      -- price or rate
        );

        CREATE descibe inverter_data (
            product_id BIGINT PRIMARY KEY,
            SKU VARCHAR(50),
            Manufacturer VARCHAR(100),
            Capacity DECIMAL(5,2),
            Weight DECIMAL(6,2),
            Dimensions VARCHAR(50),
            Rate DECIMAL(10,2)
        );

        CREATE descibe earthing_material_and_lightning_arrestor (
            product_id BIGINT PRIMARY KEY,
            SKU VARCHAR(50),
            Manufacturer VARCHAR(255),
            Length DECIMAL(6,2),
            Weight DECIMAL(6,2),
            Rate DECIMAL(10,2)
        );

        CREATE descibe earthing_rod (
            product_id BIGINT PRIMARY KEY,
            SKU VARCHAR(50),
            Length DECIMAL(6,2),
            Weight DECIMAL(6,2),
            Rate DECIMAL(10,2)
        );

        From this user input: "{user_input}", extract and validate:
        1. Manufacturer name from solar_products table. for example:  Waaree,Vikram etc.
        2. Project capacity in kW as a float number. for example: 30kW → 30.0, 40KW → 40.0, 20kw → 20.0,20Kw → 20.0, 30 kilowatt→ 30.0, etc
        3. Inverter manufacturer from inverter_data table (if explicitly mentioned)
        4. Earthing material and lightning Arrestor manufacturer from Earthing_Material_And_Lightning_Arrestor table (if explicitly mentioned)
        5. Earthing rod manufacturer from Earthing_rod table, using the SKU column (if explicitly mentioned)

        Use these tables:
        - solar_products: column name 'Make' is manufacturer and module details.
        - inverter_data: column name 'Manufacturer' is inverter.
        - Earthing_Material_And_Lightning_Arrestor: column name 'Manufacturer' contains earthing material and lightning Arrestor manufacturer names
        - Earthing_rod: column name 'SKU' contains earthing rod manufacturer names.

        Return format: {{'manufacturer': str or None, 'capacity': float or None, 'inverter': str or None, 'earthing_material': str or None, 'earthing_rod': str or None}}

        Example outputs:
        "Waaree 40Kw or Waaree 40kw or Waaree 40KW or waaree 40Kw or waaree 40kw or waaree 40KW" → {{'manufacturer': 'Waaree', 'capacity': 40.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "waaree40kw or waaree40Kw or waaree40KW or Waaree40Kw or Waaree40kw or Waaree40KW or WAAREE40Kw or WAAREE40kw or WAAREE40KW" → {{'manufacturer': 'Waaree', 'capacity': 40.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "Vikram 40Kw or Vikram 40kw or Vikram 40KW or vikram 40Kw or vikram 40kw or vikram 40KW" → {{'manufacturer': 'Vikram', 'capacity': 40.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "vikram40kw or vikram40Kw or vikram40KW or Vikram40Kw or Vikram40kw or Vikram40KW or VIKRAM40Kw or VIKRAM40kw or VIKRAM40KW" → {{'manufacturer': 'Vikram', 'capacity': 40.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "30kw Waaree system with Growatt inverter" → {{'manufacturer': 'Waaree', 'capacity': 30.0, 'inverter': 'Growatt', 'earthing_material': None, 'earthing_rod': None}}
        "Want ab boq for Vikram 50kW" → {{'manufacturer': 'Vikram', 'capacity': 50.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "Can I get boq for 70kw vikram system with Growatt inverter" → {{'manufacturer': 'Vikram', 'capacity': 70.0, 'inverter': 'Growatt', 'earthing_material': None, 'earthing_rod': None}}
        "what materials or things or products I require or need for 10kw waaree project" → {{'manufacturer': 'Waaree', 'capacity': 10.0, 'inverter': None, 'earthing_material': None, 'earthing_rod': None}}
        "Can you generate a boq for waaree with capacity 70kw" → {{'manufacturer': 'Waaree', 'capacity': 70.0, 'inverter': 'Growatt', 'earthing_material': None, 'earthing_rod': None}}
        "Generate a boq for waaree30kw solar project" → {{'manufacturer': 'Waaree', 'capacity': 30.0, 'inverter': 'Growatt', 'earthing_material': None, 'earthing_rod': None}}
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            
            # Extract the dictionary string from the response
            response_text = str(result['output'])
            dict_start = response_text.find('{')
            dict_end = response_text.rfind('}') + 1
            
            if dict_start != -1 and dict_end != -1:
                dict_str = response_text[dict_start:dict_end]
                extracted_info = ast.literal_eval(dict_str)
                
                # Additional capacity extraction if needed
                if extracted_info['capacity'] is None:
                    capacity_query = f"""
                    Extract just the project capacity in kW from: "{user_input}"
                    Return the numeric value as a float or None if not found.
                    """
                    capacity_result = self.llm.predict(capacity_query)
                    try:
                        extracted_info['capacity'] = float(capacity_result.strip())
                    except ValueError:
                        extracted_info['capacity'] = None
                
                return extracted_info
            else:
                raise ValueError("Could not find valid dictionary in response")
                
        except Exception as e:
            print(f"Error extracting information: {str(e)}")
            return {
                'manufacturer': None,
                'capacity': None,
                'inverter': None,
                'earthing_material': None,
                'earthing_rod': None
            }
    def get_module_data(self, manufacturer: str = None):
        """Get module data from MySQL database"""
        return get_module_data(self.engine, manufacturer)

    def get_inverter_data(self):
        """Get inverter data from MySQL database"""
        return get_inverter_data(self.engine)

    def get_available_manufacturers(self) -> List[str]:
        """Get list of available manufacturers"""
        return get_available_manufacturers(self.engine)

    def get_available_inverters(self) -> List[str]:
        """Get list of available inverter manufacturers"""
        return get_available_inverters(self.engine)

    def select_best_pv_module(self, manufacturer: str) -> Dict:
        """Select the best PV module for given manufacturer"""
        modules = self.get_module_data(manufacturer)
        if modules.empty:
            return None
            
        # Convert Pmax and Vmp to numeric, replacing invalid values with 0
        modules['Pmax'] = pd.to_numeric(modules['Pmax'], errors='coerce').fillna(0)
        modules['Vmp'] = pd.to_numeric(modules['Vmp'], errors='coerce').fillna(0)
        
        return modules.sort_values(by=['Pmax', 'Vmp'], 
                                ascending=[False, False]).iloc[0].to_dict()

    def get_available_pv_modules(self, manufacturer: str) -> List[Dict]:
        """Get available PV modules for given manufacturer"""
        modules = self.get_module_data(manufacturer).to_dict('records')
        
        # Convert Pmax to float and handle None/invalid values
        for module in modules:
            try:
                module['Pmax'] = float(module['Pmax']) if module['Pmax'] is not None else 0.0
            except (ValueError, TypeError):
                module['Pmax'] = 0.0
        
        return sorted(modules, key=lambda x: x['Pmax'], reverse=True)
    