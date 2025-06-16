import threading
import queue
import random
import time
import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cartel_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === DATABASE SETUP WITH ERROR HANDLING ===
def setup_database():
    """Initialize database with all required tables and handle any setup errors."""
    try:
        conn = sqlite3.connect("cartel_simulation.db", timeout=30)
        cursor = conn.cursor()

        # Create tables with error handling
        tables = {
            "ProductionEvents": """
                CREATE TABLE IF NOT EXISTS ProductionEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    location TEXT NOT NULL,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "ProcessingEvents": """
                CREATE TABLE IF NOT EXISTS ProcessingEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    location TEXT NOT NULL,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "DeliveryEvents": """
                CREATE TABLE IF NOT EXISTS DeliveryEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    truck_id TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    cargo_amount INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    delivery_time FLOAT,
                    failure_reason TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "LawEnforcementEvents": """
                CREATE TABLE IF NOT EXISTS LawEnforcementEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    impact TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "UpgradeEvents": """
                CREATE TABLE IF NOT EXISTS UpgradeEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    upgrade_type TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    description TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "SabotageEvents": """
                CREATE TABLE IF NOT EXISTS SabotageEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    sabotage_type TEXT NOT NULL,
                    damage_estimate FLOAT,
                    success BOOLEAN NOT NULL,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "MoraleEvents": """
                CREATE TABLE IF NOT EXISTS MoraleEvents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    impact FLOAT NOT NULL,
                    current_morale FLOAT NOT NULL,
                    FOREIGN KEY (simulation_id) REFERENCES SimulationRuns(id)
                )""",
            "SimulationRuns": """
                CREATE TABLE IF NOT EXISTS SimulationRuns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    duration FLOAT,
                    final_funds FLOAT,
                    final_morale FLOAT,
                    status TEXT
                )"""
        }

        for table_name, table_sql in tables.items():
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(table_sql)
                logger.info(f"Successfully created table: {table_name}")
            except sqlite3.Error as e:
                logger.error(f"Error creating table {table_name}: {e}")
                raise

        conn.commit()
        return conn

    except sqlite3.Error as e:
        logger.error(f"Database setup failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}")
        raise

# === CONFIGURATION ===
class Config:
    NUM_FARMERS = 20
    NUM_CHEMISTS = 10
    NUM_LABS = 3
    SIMULATION_TIME_LIMIT = 120
    INITIAL_LEAF_STORAGE_CAPACITY = 40
    COCAINE_STORAGE_CAPACITY = 50
    INITIAL_CARTEL_FUNDS = 1700
    FARMER_FEE = 5
    CHEMIST_FEE = 10
    BANK_LOAN_LIMIT = 2000
    STORAGE_EXPANSION_COST = 50
    STORAGE_EXPANSION_SIZE = 10
    DAY_DURATION = 5  # 5 seconds per day
    TOTAL_DAYS = 10   # Run for 10 days

TRANSPORT_TYPES = [
    {"name": "Plane", "capacity": 15, "speed": 10, "cost": 500, "routes": [
        {"name": "Route A", "revenue_per_unit": 250, "risk": 0.2},
        {"name": "Route B", "revenue_per_unit": 220, "risk": 0.15},
        {"name": "Route C", "revenue_per_unit": 270, "risk": 0.25},
        {"name": "Route D", "revenue_per_unit": 300, "risk": 0.3}
    ]},
    {"name": "Boat", "capacity": 30, "speed": 25, "cost": 300, "routes": [
        {"name": "Harbor A", "revenue_per_unit": 180, "risk": 0.1},
        {"name": "Harbor B", "revenue_per_unit": 200, "risk": 0.12},
        {"name": "Harbor C", "revenue_per_unit": 190, "risk": 0.08},
        {"name": "Harbor D", "revenue_per_unit": 210, "risk": 0.2}
    ]},
    {"name": "Truck", "capacity": 10, "speed": 20, "cost": 100, "routes": [
        {"name": "Border A", "revenue_per_unit": 150, "risk": 0.05},
        {"name": "Border B", "revenue_per_unit": 160, "risk": 0.07},
        {"name": "Border C", "revenue_per_unit": 140, "risk": 0.03},
        {"name": "Border D", "revenue_per_unit": 170, "risk": 0.09}
    ]}
]

# === SHARED RESOURCES ===
class SharedResources:
    def __init__(self):
        self.leaf_storage_capacity = Config.INITIAL_LEAF_STORAGE_CAPACITY
        self.leaf_storage = queue.Queue(maxsize=self.leaf_storage_capacity)
        self.cocaine_storage = {"quantity": 0, "lock": threading.Condition()}
        self.cartel_funds = Config.INITIAL_CARTEL_FUNDS
        self.cartel_loan = 0
        self.cartel_funds_lock = threading.Lock()
        self.overflow_penalty = 0
        self.simulation_id = None
        self.simulation_running = True

# === FACILITY STATE PATTERN ===
class FacilityState(ABC):
    @abstractmethod
    def can_operate(self): pass
    
    @abstractmethod
    def handle_operation(self, lab, item): pass

class RunningState(FacilityState):
    def can_operate(self): return True
    def handle_operation(self, lab, item): return True

class MaintenanceState(FacilityState):
    def can_operate(self): 
        logger.info(f"Lab {lab.lab_id} is under maintenance.")
        return False
    def handle_operation(self, lab, item): return False

class CompromisedState(FacilityState):
    def can_operate(self): return True
    def handle_operation(self, lab, item): 
        logger.warning(f"Lab {lab.lab_id} destroyed {item}!")
        return False

class Lab:
    def __init__(self, lab_id, db_connection_factory):
        self.lab_id = lab_id
        self.is_operational = True
        self.state = RunningState()
        self.db_connection_factory = db_connection_factory

    def update_state(self):
        roll = random.random()
        if roll < 0.05:
            self.state = MaintenanceState()
            self.is_operational = False
            self.log_state_change("Maintenance")
        elif roll < 0.08:  # 3% chance of being compromised
            self.state = CompromisedState()
            self.is_operational = False
            self.log_state_change("Compromised")
        else:
            self.state = RunningState()
            self.is_operational = True

    def log_state_change(self, new_state):
        try:
            conn = self.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "State Change",
                f"Lab {self.lab_id} changed to {new_state}",
                False if new_state != "Running" else True,
                f"Lab-{self.lab_id}"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging lab state change: {e}")

    def can_operate(self):
        if not self.is_operational:
            logger.info(f"Lab {self.lab_id} cannot operate (under maintenance/compromised).")
            return False
        return True

    def handle_operation(self, item):
        if not self.can_operate():
            logger.warning(f"Lab {self.lab_id} failed to handle operation for {item}.")
            return False
        return self.state.handle_operation(self, item)

# === OBSERVER PATTERN FOR MORALE ===
class MoraleObserver(ABC):
    @abstractmethod
    def update(self, morale): pass

class MoraleSubject:
    def __init__(self):
        self._observers = []
        self._lock = threading.Lock()

    def attach(self, observer):
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def detach(self, observer):
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def notify_observers(self, morale):
        with self._lock:
            for observer in self._observers:
                try:
                    observer.update(morale)
                except Exception as e:
                    logger.error(f"Error notifying observer: {e}")

class CartelMorale(MoraleSubject, threading.Thread):
    def __init__(self, chaotic_factor=0.5):
        MoraleSubject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        self.morale = 1.0  # Initial morale (1.0 = 100%)
        self.chaotic_factor = chaotic_factor
        self.running = True
        self.db_connection_factory = None

    def set_db_connection_factory(self, factory):
        self.db_connection_factory = factory

    def run(self):
        while self.running and shared_resources.simulation_running:
            try:
                if self.chaotic_factor > 0.7:
                    self.trigger_consequence()
                self.notify_observers(self.morale)
                self.log_morale()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in morale system: {e}")
                break

    def trigger_consequence(self):
        chance = random.random() * self.chaotic_factor
        if chance > 0.7:
            consequence = random.choice(["mass_desertion", "internal_conflict", "paranoia"])
            if consequence == "mass_desertion":
                impact = -0.15
                desc = "Mass desertion occurred! Morale drops sharply."
            elif consequence == "internal_conflict":
                impact = -0.1
                desc = "Internal conflict occurred! Morale drops moderately."
            else:  # paranoia
                impact = -0.05
                desc = "Paranoia spread! Morale dips slightly."
            
            self.morale = max(0, self.morale + impact)
            self.log_morale_event(consequence, desc, impact)

    def log_morale(self):
        if not self.db_connection_factory:
            return
            
        try:
            conn = self.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO MoraleEvents (simulation_id, timestamp, event_type, description, impact, current_morale)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "Update",
                "Regular morale update",
                0,
                self.morale
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging morale: {e}")

    def log_morale_event(self, event_type, description, impact):
        if not self.db_connection_factory:
            return
            
        try:
            conn = self.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO MoraleEvents (simulation_id, timestamp, event_type, description, impact, current_morale)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                event_type,
                description,
                impact,
                self.morale
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging morale event: {e}")

    def stop(self):
        self.running = False

# === CHAIN OF RESPONSIBILITY ===
class Handler(ABC):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler):
        self.next_handler = handler
        return handler

    @abstractmethod
    def handle(self, item): pass

class QualityAssessor(Handler):
    def handle(self, item):
        try:
            quality = random.choice(['Low', 'Medium', 'High'])
            logger.info(f"Assessed {item} as {quality} quality.")
            
            if quality == 'Low':
                logger.info(f"Discarded {item} for low quality.")
                self.log_quality_assessment(item, quality, False)
                return None
                
            self.log_quality_assessment(item, quality, True)
            return super().handle((item, quality))
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return None

    def log_quality_assessment(self, item, quality, success):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "Quality Assessment",
                f"Assessed {item} as {quality} quality",
                success,
                "Quality Control"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging quality assessment: {e}")

class StorageManager(Handler):
    def handle(self, item):
        try:
            shared_resources.leaf_storage.put(item, block=False)
            logger.info(f"Stored {item[0]} ({item[1]}) in storage.")
            self.log_storage_event(item[0], item[1], True)
            return item
        except queue.Full:
            with shared_resources.cartel_funds_lock:
                if shared_resources.cartel_funds >= Config.STORAGE_EXPANSION_COST:
                    shared_resources.cartel_funds -= Config.STORAGE_EXPANSION_COST
                    shared_resources.leaf_storage_capacity += Config.STORAGE_EXPANSION_SIZE
                    shared_resources.leaf_storage = queue.Queue(maxsize=shared_resources.leaf_storage_capacity)
                    logger.info(f"Storage expanded to {shared_resources.leaf_storage_capacity}.")
                    self.log_storage_expansion()
                    return self.handle(item)  # Retry after expansion
                else:
                    logger.warning(f"Storage full. {item[0]} lost.")
                    self.log_storage_event(item[0], item[1], False)
                    return None
        except Exception as e:
            logger.error(f"Error in storage management: {e}")
            return None

    def log_storage_event(self, item, quality, success):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "Storage",
                f"Stored {item} ({quality})",
                success,
                "Storage"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging storage event: {e}")

    def log_storage_expansion(self):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO UpgradeEvents (simulation_id, timestamp, upgrade_type, target_entity, description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "Storage Expansion",
                "Leaf Storage",
                f"Expanded to {shared_resources.leaf_storage_capacity} capacity"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging storage expansion: {e}")

# === STRATEGY PATTERN FOR ROUTES ===
class RoutingStrategy(ABC):
    @abstractmethod
    def select_route(self, routes): pass

class HighRevenueStrategy(RoutingStrategy):
    def select_route(self, routes):
        return max(routes, key=lambda r: r['revenue_per_unit'])

class LowRiskStrategy(RoutingStrategy):
    def select_route(self, routes):
        return min(routes, key=lambda r: r['risk'])

class BalancedStrategy(RoutingStrategy):
    def select_route(self, routes):
        return max(routes, key=lambda r: r['revenue_per_unit'] / (r['risk'] + 0.01))

# === WORKER CLASSES ===
class Farmer(threading.Thread, MoraleObserver):
    def __init__(self, farmer_id, chain):
        threading.Thread.__init__(self, daemon=True)
        self.farmer_id = farmer_id
        self.chain = chain
        self.efficiency = 1.0
        self.running = True

    def update(self, morale):
        """Update the farmer's efficiency based on morale."""
        try:
            if morale <= 0:
                logger.warning(f"Warning: Morale is {morale}. Setting efficiency to minimum value.")
                self.efficiency = 0.1  # Set a minimum efficiency value
            else:
                self.efficiency = morale
        except Exception as e:
            logger.error(f"Farmer {self.farmer_id} error updating morale: {e}")
            self.efficiency = 0.5  # Default value on error

    def log_production_event(self, event_type, description, quantity):
        """Logs a production event to the ProductionEvents table."""
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProductionEvents (simulation_id, timestamp, event_type, description, quantity, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                event_type,
                description,
                quantity,
                f"Farmer-{self.farmer_id}"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging production event: {e}")

    def run(self):
        while self.running and shared_resources.simulation_running:
            try:
                time.sleep(max(0.1, random.uniform(2, 5) / max(0.1, self.efficiency)))
                
                if random.random() < 0.15 * (1.1 - self.efficiency):
                    logger.warning(f"Farmer {self.farmer_id}: Crop failed.")
                    self.log_production_event("Crop Failure", "Crop failed due to low morale", 0)
                    continue
                    
                item = f"Leaf-{self.farmer_id}"
                logger.info(f"Farmer {self.farmer_id}: Successfully harvested {item}.")
                self.log_production_event("Harvest Success", f"Harvested {item}", 1)
                
                result = self.chain.handle(item)
                if result is None:
                    logger.warning(f"Farmer {self.farmer_id}: Item {item} was rejected by the chain.")
                    
            except Exception as e:
                logger.error(f"Farmer {self.farmer_id} encountered error: {e}")
                time.sleep(5)  # Prevent tight error loop

    def stop(self):
        self.running = False

class Chemist(threading.Thread, MoraleObserver):
    def __init__(self, chemist_id, lab):
        threading.Thread.__init__(self, daemon=True)
        self.chemist_id = chemist_id
        self.lab = lab
        self.efficiency = 1.0
        self.running = True

    def update(self, morale):
        """Update the chemist's efficiency based on morale."""
        try:
            self.efficiency = max(0.1, morale)  # Ensure minimum efficiency
        except Exception as e:
            logger.error(f"Chemist {self.chemist_id} error updating morale: {e}")
            self.efficiency = 0.5  # Default value on error

    def log_processing_event(self, event_type, description, success):
        """Logs an event to the ProcessingEvents table."""
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                event_type,
                description,
                success,
                f"Chemist-{self.chemist_id}"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging processing event: {e}")

    def run(self):
        while self.running and shared_resources.simulation_running:
            try:
                leaves, quality = shared_resources.leaf_storage.get(timeout=5)
                
                if random.random() < 0.1 * (1.1 - self.efficiency):
                    logger.warning(f"Chemist {self.chemist_id}: Raided! Lost {leaves}.")
                    self.log_processing_event("Processing Failure", f"Failed to process {leaves}", False)
                    continue
                    
                self.lab.update_state()
                if not self.lab.can_operate():
                    self.log_processing_event("Processing Failure", f"Lab {self.lab.lab_id} is under maintenance", False)
                    continue
                    
                if not self.lab.handle_operation(leaves):
                    self.log_processing_event("Processing Failure", f"Lab {self.lab.lab_id} failed to handle {leaves}", False)
                    continue
                    
                with shared_resources.cocaine_storage["lock"]:
                    if shared_resources.cocaine_storage["quantity"] < Config.COCAINE_STORAGE_CAPACITY:
                        shared_resources.cocaine_storage["quantity"] += 1
                        logger.info(f"Chemist {self.chemist_id} processed {leaves} -> Total: {shared_resources.cocaine_storage['quantity']}")
                        self.log_processing_event("Processing Success", f"Processed {leaves}", True)
                        shared_resources.cocaine_storage["lock"].notify_all()
                    else:
                        shared_resources.overflow_penalty += 1
                        logger.warning(f"Chemist {self.chemist_id}: Storage full. Discarded {leaves}.")
                        self.log_processing_event("Processing Failure", f"Storage full, discarded {leaves}", False)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Chemist {self.chemist_id} encountered error: {e}")
                time.sleep(5)  # Prevent tight error loop

    def stop(self):
        self.running = False

class Distributor(threading.Thread, MoraleObserver):
    def __init__(self, distributor_id, name, capacity, speed, cost, routes, strategy):
        threading.Thread.__init__(self, daemon=True)
        self.distributor_id = distributor_id
        self.name = name
        self.capacity = capacity
        self.speed = speed
        self.cost = cost
        self.routes = routes
        self.strategy = strategy
        self.efficiency = 1.0
        self.running = True

    def update(self, morale):
        """Update the distributor's efficiency based on morale."""
        try:
            self.efficiency = max(0.1, morale)  # Ensure minimum efficiency
        except Exception as e:
            logger.error(f"Distributor {self.distributor_id} error updating morale: {e}")
            self.efficiency = 0.5  # Default value on error

    def log_delivery_event(self, event_type, description, success, route_name=None, revenue=0):
        """Logs an event to the DeliveryEvents table."""
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO DeliveryEvents (
                    simulation_id, timestamp, truck_id, origin, destination, 
                    cargo_amount, success, delivery_time, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                f"{self.name}-{self.distributor_id}",
                "Lab",
                route_name if route_name else "Unknown",
                self.capacity,
                success,
                self.speed / max(0.1, self.efficiency),
                None if success else description
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging delivery event: {e}")

    def run(self):
        while self.running and shared_resources.simulation_running:
            try:
                with shared_resources.cocaine_storage["lock"]:
                    while shared_resources.cocaine_storage["quantity"] < self.capacity:
                        if not shared_resources.simulation_running:
                            return
                        shared_resources.cocaine_storage["lock"].wait(1)
                    shared_resources.cocaine_storage["quantity"] -= self.capacity
                    
                route = self.strategy.select_route(self.routes)
                delivery_time = max(0.1, self.speed / max(0.1, self.efficiency))
                time.sleep(delivery_time)
                
                revenue = self.capacity * route['revenue_per_unit']
                if random.random() < route['risk'] * (1.1 - self.efficiency):
                    if random.random() < 0.5:
                        logger.warning(f"{self.name} {self.distributor_id}: Bribe successful on {route['name']}.")
                        revenue *= 0.8
                        self.log_delivery_event("Shipment Delivered", f"Bribe successful on {route['name']}", True, route['name'], revenue)
                    else:
                        logger.warning(f"{self.name} {self.distributor_id}: Shipment seized on {route['name']}.")
                        revenue = 0
                        self.log_delivery_event("Shipment Failure", f"Shipment seized on {route['name']}", False, route['name'], 0)
                        self.log_law_enforcement_event("Seizure", f"Shipment seized on {route['name']}", route['name'])
                else:
                    logger.info(f"{self.name} {self.distributor_id} delivered via {route['name']}. Revenue: ${revenue}")
                    self.log_delivery_event("Shipment Delivered", f"Delivered via {route['name']}", True, route['name'], revenue)
                    
                with shared_resources.cartel_funds_lock:
                    if shared_resources.cartel_funds < self.cost:
                        loan = self.cost - shared_resources.cartel_funds
                        if shared_resources.cartel_loan + loan <= Config.BANK_LOAN_LIMIT:
                            shared_resources.cartel_loan += loan
                            shared_resources.cartel_funds += loan
                            logger.info(f"Loan taken: ${loan}.")
                            self.log_financial_event("Loan Taken", f"Took loan of ${loan}")
                    shared_resources.cartel_funds -= self.cost
                    shared_resources.cartel_funds += revenue
                    self.log_financial_event("Revenue", f"Earned ${revenue} from delivery")
                    
            except Exception as e:
                logger.error(f"Distributor {self.distributor_id} encountered error: {e}")
                time.sleep(5)  # Prevent tight error loop

    def log_law_enforcement_event(self, event_type, description, location):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO LawEnforcementEvents (simulation_id, timestamp, event_type, location, impact)
                VALUES (?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                event_type,
                location,
                description
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging law enforcement event: {e}")

    def log_financial_event(self, event_type, description):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                event_type,
                description,
                True,
                "Financial"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging financial event: {e}")

    def stop(self):
        self.running = False

# === PAYROLL SYSTEM ===
class Payroll(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.running = True

    def log_payroll_event(self, amount, success):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessingEvents (simulation_id, timestamp, event_type, description, success, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                "Payroll",
                f"Payroll processed: ${amount}",
                success,
                "Payroll"
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging payroll event: {e}")

    def run(self):
        while self.running and shared_resources.simulation_running:
            try:
                time.sleep(20)  # Pay every 20 seconds
                with shared_resources.cartel_funds_lock:
                    total_pay = Config.NUM_FARMERS * Config.FARMER_FEE + Config.NUM_CHEMISTS * Config.CHEMIST_FEE
                    if shared_resources.cartel_funds < total_pay:
                        need = total_pay - shared_resources.cartel_funds
                        if shared_resources.cartel_loan + need <= Config.BANK_LOAN_LIMIT:
                            shared_resources.cartel_loan += need
                            shared_resources.cartel_funds += need
                            logger.info(f"Loan for payroll: ${need}")
                            self.log_payroll_event(need, True)
                        else:
                            logger.warning(f"Insufficient funds for payroll. Missing: ${need}")
                            self.log_payroll_event(total_pay, False)
                            continue
                    shared_resources.cartel_funds -= total_pay
                    logger.info(f"Paid payroll: ${total_pay}. Remaining funds: ${shared_resources.cartel_funds}")
                    self.log_payroll_event(total_pay, True)
            except Exception as e:
                logger.error(f"Payroll error: {e}")
                time.sleep(5)  # Prevent tight error loop

    def stop(self):
        self.running = False

# === SABOTAGE SYSTEM ===
class SabotageStrategy(ABC):
    @abstractmethod
    def execute(self, target, cartel_morale): pass

class DirectAttackStrategy(SabotageStrategy):
    def execute(self, target, cartel_morale):
        result = None
        if target == "Truck":
            if random.random() < 0.6:  # 60% chance of hijack
                cargo_loss = random.randint(50, 100)  # 50–100% cargo loss
                result = (f"Truck hijacked! Lost {cargo_loss}% of cargo.", True, cargo_loss/100)
        elif target == "Storage":
            if random.random() < 0.4:  # 40% chance of raid
                goods_stolen = random.randint(20, 50)  # 20–50% goods stolen
                result = (f"Storage raided! Lost {goods_stolen}% of goods.", True, goods_stolen/100)
        elif target == "Personnel":
            if random.random() < 0.7:  # 70% chance of intimidation
                efficiency_loss = 0.3  # 30% efficiency loss
                cartel_morale.morale -= efficiency_loss
                result = (f"Personnel intimidated! Efficiency reduced by {efficiency_loss*100}%.", True, efficiency_loss)
        return result

class SubversiveStrategy(SabotageStrategy):
    def execute(self, target, cartel_morale):
        result = None
        if target == "Chemists" or target == "Farmers":
            if random.random() < 0.5:  # 50% chance of corruption
                result = (f"{target} corrupted! Skill/reputation reduced.", True, 0.2)
        elif target == "Reputation":
            if random.random() < 0.7:  # 70% chance
                result = ("Reputation hit! Market trust reduced.", True, 0.15)
        elif target == "Quality":
            if random.random() < 0.4:  # 40% chance
                result = ("Product contaminated! Quality reduced.", True, 0.25)
        elif target == "Morale":
            if random.random() < 0.6:  # 60% chance
                morale_drop = 0.2
                cartel_morale.morale -= morale_drop
                result = ("Fake news spread! Morale dropped.", True, morale_drop)
        return result

class MarketDisruptionStrategy(SabotageStrategy):
    def execute(self, target, cartel_morale):
        result = None
        if target == "Route":
            if random.random() < 0.5:  # 50% chance of disruption
                result = ("Transport route disrupted! Delivery affected.", True, 0.3)
        elif target == "Price":
            if random.random() < 0.6:  # 60% chance
                result = ("Market flooded! Prices dropped.", True, 0.4)
        elif target == "Law Enforcement":
            if random.random() < 0.4:  # 40% chance
                result = ("Police tipped off! Future raids expected.", True, 0.35)
        return result

class SabotageManager:
    def __init__(self, cartel_morale):
        self.cartel_morale = cartel_morale
        self.strategies = {
            "DirectAttack": DirectAttackStrategy(),
            "Subversive": SubversiveStrategy(),
            "MarketDisruption": MarketDisruptionStrategy()
        }

    def execute_strategy(self, strategy_name, target):
        try:
            strategy = self.strategies.get(strategy_name)
            if strategy:
                result = strategy.execute(target, self.cartel_morale)
                if result:
                    description, success, impact = result
                    logger.warning(f"Sabotage: {description}")
                    self.log_sabotage_event(target, strategy_name, description, success, impact)
        except Exception as e:
            logger.error(f"Error executing sabotage strategy: {e}")

    def log_sabotage_event(self, target, strategy, description, success, impact):
        try:
            conn = shared_resources.db_connection_factory()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO SabotageEvents (
                    simulation_id, timestamp, target_type, target_id, 
                    sabotage_type, damage_estimate, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                shared_resources.simulation_id,
                datetime.now().isoformat(),
                target,
                "N/A",
                strategy,
                impact,
                success
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error logging sabotage event: {e}")

# === SIMULATION CONTROL ===
def initialize_simulation(db_conn):
    """Initialize the simulation and return simulation ID."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO SimulationRuns (start_time, status)
            VALUES (?, ?)
        """, (datetime.now().isoformat(), "Running"))
        db_conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Error initializing simulation: {e}")
        raise

def finalize_simulation(db_conn, simulation_id, status, final_funds, final_morale):
    """Finalize the simulation with end stats."""
    try:
        cursor = db_conn.cursor()
        # Get the start time first
        cursor.execute("SELECT start_time FROM SimulationRuns WHERE id = ?", (simulation_id,))
        start_time = datetime.fromisoformat(cursor.fetchone()[0])
        duration = (datetime.now() - start_time).total_seconds()
        
        cursor.execute("""
            UPDATE SimulationRuns 
            SET end_time = ?, duration = ?, final_funds = ?, final_morale = ?, status = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            duration,
            final_funds,
            final_morale,
            status,
            simulation_id
        ))
        db_conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error finalizing simulation: {e}")
        raise

def preview_all_tables(db_path="cartel_simulation.db"):
    """Preview all tables in the database by printing their contents."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        logger.info("\n=== Database Tables Preview ===")
        for table_name in tables:
            table_name = table_name[0]
            logger.info(f"\n--- {table_name} ---")
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            if rows:
                for row in rows:
                    logger.info(row)
            else:
                logger.info("No data available.")

        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error while previewing tables: {e}")

# === MAIN SIMULATION ===
def main():
    global shared_resources
    
    try:
        # Initialize database and shared resources
        db_conn = setup_database()
        shared_resources = SharedResources()
        shared_resources.db_connection_factory = lambda: sqlite3.connect("cartel_simulation.db", timeout=30)
        
        # Initialize simulation run
        shared_resources.simulation_id = initialize_simulation(db_conn)
        logger.info(f"Simulation started with ID: {shared_resources.simulation_id}")
        
        # Initialize morale system
        cartel_morale = CartelMorale(chaotic_factor=0.8)
        cartel_morale.set_db_connection_factory(shared_resources.db_connection_factory)
        
        # Initialize chain of responsibility
        qa = QualityAssessor()
        sm = StorageManager()
        qa.set_next(sm)
        
        # Initialize labs
        labs = [Lab(i, shared_resources.db_connection_factory) for i in range(Config.NUM_LABS)]
        
        # Initialize workers
        farmers = [Farmer(i, qa) for i in range(Config.NUM_FARMERS)]
        chemists = [Chemist(i, random.choice(labs)) for i in range(Config.NUM_CHEMISTS)]
        
        # Initialize distributors with strategies
        strategies = [HighRevenueStrategy(), LowRiskStrategy(), BalancedStrategy()]
        distributors = [
        Distributor(
            i,
            name=t["name"],
            capacity=t["capacity"],
            speed=t["speed"],
            cost=t["cost"],
            routes=t["routes"],
            strategy=random.choice(strategies))
        for i, t in enumerate(TRANSPORT_TYPES)
        ]
        
        # Attach all workers to morale system
        for worker in farmers + chemists + distributors:
            cartel_morale.attach(worker)
            
        # Start systems
        payroll = Payroll()
        sabotage_manager = SabotageManager(cartel_morale)
        
        # Start all threads
        cartel_morale.start()
        payroll.start()
        for worker in farmers + chemists + distributors:
            worker.start()
            
        # Main simulation loop
        start_time = time.time()
        current_day = 0
        
        while current_day < Config.TOTAL_DAYS and shared_resources.simulation_running:
            day_start_time = time.time()
            current_day += 1
            logger.info(f"\n=== Starting day {current_day} ===")
            
            while time.time() - day_start_time < Config.DAY_DURATION and shared_resources.simulation_running:
                time.sleep(1)
                # Execute random sabotage events
                if random.random() < 0.3:  # 30% chance per day
                    strategy = random.choice(["DirectAttack", "Subversive", "MarketDisruption"])
                    target = random.choice(["Truck", "Storage", "Personnel", "Chemists", "Farmers", "Reputation", "Quality", "Morale", "Route", "Price", "Law Enforcement"])
                    sabotage_manager.execute_strategy(strategy, target)
                    
            logger.info(f"=== Day {current_day} ended ===")
            
            # Check for catastrophic failure conditions
            if cartel_morale.morale <= 0.1:
                logger.error("Critical morale failure! Simulation ending early.")
                shared_resources.simulation_running = False
                break
                
            if shared_resources.cartel_loan > Config.BANK_LOAN_LIMIT * 1.5:
                logger.error("Critical financial failure! Simulation ending early.")
                shared_resources.simulation_running = False
                break
                
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
        shared_resources.simulation_running = False
    except Exception as e:
        logger.error(f"Critical simulation error: {e}")
        shared_resources.simulation_running = False
    finally:
        # Cleanup
        logger.info("Stopping all systems...")
        cartel_morale.stop()
        payroll.stop()
        
        for worker in farmers + chemists + distributors:
            worker.stop()
            
        # Wait for threads to finish
        for worker in farmers + chemists + distributors:
            worker.join(timeout=1)
            
        # Finalize simulation
        final_status = "Completed" if current_day >= Config.TOTAL_DAYS else "Aborted"
        finalize_simulation(
            db_conn,
            shared_resources.simulation_id,
            final_status,
            shared_resources.cartel_funds,
            cartel_morale.morale
        )
        
        # Print final stats
        logger.info("\n=== Simulation Results ===")
        logger.info(f"Final funds: ${shared_resources.cartel_funds}")
        logger.info(f"Final loan amount: ${shared_resources.cartel_loan}")
        logger.info(f"Final morale: {cartel_morale.morale:.2f}")
        logger.info(f"Total days simulated: {current_day}")
        
        # Preview database
        preview_all_tables()
        
        # Close database connection
        db_conn.close()
        logger.info("Simulation complete.")

if __name__ == "__main__":
    shared_resources = None

main()