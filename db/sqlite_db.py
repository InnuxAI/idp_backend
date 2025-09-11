import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import os

class SQLiteDB:
    def __init__(self, db_path: str = "extraction_schemas.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_schemas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    json_name TEXT UNIQUE NOT NULL,
                    json_description TEXT,
                    json_string TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    extracted_data TEXT NOT NULL,
                    is_approved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (schema_id) REFERENCES extraction_schemas (id)
                )
            """)
            
            conn.commit()
    
    def save_schema(self, name: str, description: Optional[str], schema_json: Dict[Any, Any]) -> int:
        """Save a new extraction schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO extraction_schemas (json_name, json_description, json_string, created_at)
                VALUES (?, ?, ?, ?)
            """, (name, description, json.dumps(schema_json), datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all extraction schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, json_name, json_description, json_string, created_at
                FROM extraction_schemas
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            
            schemas = []
            for row in rows:
                schema = dict(row)
                schema['json_string'] = json.loads(schema['json_string'])
                schemas.append(schema)
            
            return schemas
    
    def get_schema_by_id(self, schema_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific schema by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, json_name, json_description, json_string, created_at
                FROM extraction_schemas
                WHERE id = ?
            """, (schema_id,))
            row = cursor.fetchone()
            
            if row:
                schema = dict(row)
                schema['json_string'] = json.loads(schema['json_string'])
                return schema
            return None
    
    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific schema by name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, json_name, json_description, json_string, created_at
                FROM extraction_schemas
                WHERE json_name = ?
            """, (name,))
            row = cursor.fetchone()
            
            if row:
                schema = dict(row)
                schema['json_string'] = json.loads(schema['json_string'])
                return schema
            return None
    
    def update_schema(self, schema_id: int, name: str, description: Optional[str], schema_json: Dict[Any, Any]) -> bool:
        """Update an existing schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE extraction_schemas
                SET json_name = ?, json_description = ?, json_string = ?
                WHERE id = ?
            """, (name, description, json.dumps(schema_json), schema_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_schema(self, schema_id: int) -> bool:
        """Delete a schema by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM extraction_schemas WHERE id = ?
            """, (schema_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def schema_name_exists(self, name: str, exclude_id: Optional[int] = None) -> bool:
        """Check if a schema name already exists"""
        with sqlite3.connect(self.db_path) as conn:
            if exclude_id:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM extraction_schemas 
                    WHERE json_name = ? AND id != ?
                """, (name, exclude_id))
            else:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM extraction_schemas 
                    WHERE json_name = ?
                """, (name,))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    # Data Library methods
    def save_extraction(self, schema_id: int, filename: str, pdf_path: str, extracted_data: Dict[Any, Any]) -> int:
        """Save extracted data to data library"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO data_library (schema_id, filename, pdf_path, extracted_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (schema_id, filename, pdf_path, json.dumps(extracted_data), 
                  datetime.now().isoformat(), datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
    
    def get_extractions(self, schema_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all extractions, optionally filtered by schema_id"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if schema_id:
                cursor = conn.execute("""
                    SELECT dl.*, es.json_name as schema_name
                    FROM data_library dl
                    JOIN extraction_schemas es ON dl.schema_id = es.id
                    WHERE dl.schema_id = ?
                    ORDER BY dl.created_at DESC
                """, (schema_id,))
            else:
                cursor = conn.execute("""
                    SELECT dl.*, es.json_name as schema_name
                    FROM data_library dl
                    JOIN extraction_schemas es ON dl.schema_id = es.id
                    ORDER BY dl.created_at DESC
                """)
            
            rows = cursor.fetchall()
            
            extractions = []
            for row in rows:
                extraction = dict(row)
                extraction['extracted_data'] = json.loads(extraction['extracted_data'])
                extractions.append(extraction)
            
            return extractions
    
    def get_extraction_by_id(self, extraction_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific extraction by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT dl.*, es.json_name as schema_name, es.json_string as schema_definition
                FROM data_library dl
                JOIN extraction_schemas es ON dl.schema_id = es.id
                WHERE dl.id = ?
            """, (extraction_id,))
            row = cursor.fetchone()
            
            if row:
                extraction = dict(row)
                extraction['extracted_data'] = json.loads(extraction['extracted_data'])
                extraction['schema_definition'] = json.loads(extraction['schema_definition'])
                return extraction
            return None
    
    def update_extraction(self, extraction_id: int, extracted_data: Dict[Any, Any], is_approved: bool = False) -> bool:
        """Update extraction data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE data_library
                SET extracted_data = ?, is_approved = ?, updated_at = ?
                WHERE id = ?
            """, (json.dumps(extracted_data), is_approved, datetime.now().isoformat(), extraction_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_extraction(self, extraction_id: int) -> bool:
        """Delete an extraction by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM data_library
                WHERE id = ?
            """, (extraction_id,))
            conn.commit()
            return cursor.rowcount > 0

# Global database instance
db = SQLiteDB()
