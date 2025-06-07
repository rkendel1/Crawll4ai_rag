"""
OpenAPI Processing Module for Crawl4AI RAG MCP Server
Handles parsing and chunking of OpenAPI specifications
"""

import json
import yaml
from prance import ResolvingParser
from typing import List, Dict, Optional, Any
import hashlib
from datetime import datetime
import logging
from pathlib import Path
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class OpenAPIProcessor:
    """Processes OpenAPI specifications for ingestion into vector database"""
    
    def __init__(self):
        """Initialize OpenAPI processor"""
        self.chunk_strategies = {
            "endpoint": self._chunk_by_endpoint,
            "schema": self._chunk_by_schema,
            "combined": self._chunk_combined,
            "operation": self._chunk_by_operation
        }
    
    async def process_openapi(self, file_path: str, 
                            strategy: str = "combined") -> List[Dict]:
        """
        Parse and chunk OpenAPI specification
        
        Args:
            file_path: Path to OpenAPI spec file (JSON or YAML)
            strategy: Chunking strategy - 'endpoint', 'schema', 'combined', or 'operation'
            
        Returns:
            List of chunks with content and metadata
            
        Raises:
            FileNotFoundError: If spec file doesn't exist
            ValidationError: If spec is invalid
            Exception: For parsing errors
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"OpenAPI spec file not found: {file_path}")
        
        try:
            # Parse OpenAPI spec with reference resolution
            logger.info(f"Parsing OpenAPI spec: {path.name}")
            parser = ResolvingParser(str(path.absolute()))
            spec = parser.specification
            
            # Validate it's a valid OpenAPI spec
            if "openapi" not in spec and "swagger" not in spec:
                raise ValidationError("File does not appear to be an OpenAPI specification")
            
            # Extract metadata
            info = spec.get("info", {})
            metadata = {
                "source": str(path.absolute()),
                "filename": path.name,
                "type": "openapi",
                "openapi_version": spec.get("openapi", spec.get("swagger", "3.0.0")),
                "title": info.get("title", "Unknown API"),
                "api_version": info.get("version", "1.0.0"),
                "description": info.get("description", ""),
                "extracted_at": datetime.now().isoformat()
            }
            
            # Apply chunking strategy
            if strategy not in self.chunk_strategies:
                logger.warning(f"Unknown strategy '{strategy}', using 'combined'")
                strategy = "combined"
                
            chunk_fn = self.chunk_strategies[strategy]
            chunks = chunk_fn(spec, metadata)
            
            logger.info(f"Created {len(chunks)} chunks using '{strategy}' strategy")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing OpenAPI spec: {str(e)}")
            raise Exception(f"Failed to process OpenAPI spec: {str(e)}")
    
    def _chunk_by_endpoint(self, spec: Dict, metadata: Dict) -> List[Dict]:
        """Create chunks for each endpoint with full context"""
        chunks = []
        paths = spec.get("paths", {})
        
        # Get global parameters if any
        global_params = spec.get("parameters", {})
        
        for path, path_item in paths.items():
            # Skip if not a path item
            if not isinstance(path_item, dict):
                continue
                
            # Get path-level parameters
            path_params = path_item.get("parameters", [])
            
            for method, operation in path_item.items():
                # Skip non-operation fields
                if method not in ["get", "post", "put", "delete", "patch", "options", "head", "trace"]:
                    continue
                
                if not isinstance(operation, dict):
                    continue
                
                # Create comprehensive endpoint documentation
                content = self._format_endpoint(path, method, operation, 
                                              path_params, spec)
                
                # Generate unique chunk ID
                chunk_id = hashlib.md5(
                    f"{path}_{method}_{metadata['title']}".encode()
                ).hexdigest()
                
                chunk_metadata = {
                    **metadata,
                    "chunk_type": "endpoint",
                    "endpoint": f"{method.upper()} {path}",
                    "operation_id": operation.get("operationId", ""),
                    "tags": operation.get("tags", []),
                    "summary": operation.get("summary", "")
                }
                
                chunks.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def _chunk_by_schema(self, spec: Dict, metadata: Dict) -> List[Dict]:
        """Create chunks for each schema/model definition"""
        chunks = []
        
        # Handle both OpenAPI 3.x and 2.x locations
        schemas = {}
        if "components" in spec and "schemas" in spec["components"]:
            schemas = spec["components"]["schemas"]
        elif "definitions" in spec:
            schemas = spec["definitions"]
        
        for schema_name, schema_def in schemas.items():
            content = f"# Schema: {schema_name}\n\n"
            
            # Add description if available
            if "description" in schema_def:
                content += f"{schema_def['description']}\n\n"
            
            # Format the schema definition
            content += "## Definition\n\n```json\n"
            content += json.dumps(schema_def, indent=2)
            content += "\n```\n\n"
            
            # Add usage examples if we can find endpoints using this schema
            usage = self._find_schema_usage(schema_name, spec)
            if usage:
                content += "## Used In\n\n"
                for endpoint in usage:
                    content += f"- {endpoint}\n"
                content += "\n"
            
            # Generate unique chunk ID
            chunk_id = hashlib.md5(
                f"schema_{schema_name}_{metadata['title']}".encode()
            ).hexdigest()
            
            chunk_metadata = {
                **metadata,
                "chunk_type": "schema",
                "schema_name": schema_name,
                "schema_type": schema_def.get("type", "object")
            }
            
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _chunk_combined(self, spec: Dict, metadata: Dict) -> List[Dict]:
        """Combine multiple chunking strategies"""
        chunks = []
        
        # Add overview chunk
        overview = self._create_overview_chunk(spec, metadata)
        if overview:
            chunks.append(overview)
        
        # Add endpoint chunks
        chunks.extend(self._chunk_by_endpoint(spec, metadata))
        
        # Add schema chunks
        chunks.extend(self._chunk_by_schema(spec, metadata))
        
        # Add security schemes if present
        security_chunks = self._chunk_security_schemes(spec, metadata)
        chunks.extend(security_chunks)
        
        return chunks
    
    def _chunk_by_operation(self, spec: Dict, metadata: Dict) -> List[Dict]:
        """Create chunks grouped by operationId"""
        chunks = []
        operations = {}
        
        # Group endpoints by operationId
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
                
            for method, operation in path_item.items():
                if method not in ["get", "post", "put", "delete", "patch", "options", "head", "trace"]:
                    continue
                    
                if not isinstance(operation, dict):
                    continue
                    
                op_id = operation.get("operationId", f"{method}_{path}")
                if op_id not in operations:
                    operations[op_id] = []
                    
                operations[op_id].append({
                    "path": path,
                    "method": method,
                    "operation": operation
                })
        
        # Create chunks for each operation group
        for op_id, endpoints in operations.items():
            content = f"# Operation: {op_id}\n\n"
            
            for ep in endpoints:
                content += self._format_endpoint(
                    ep["path"], ep["method"], ep["operation"], [], spec
                )
                content += "\n---\n\n"
            
            chunk_id = hashlib.md5(f"operation_{op_id}".encode()).hexdigest()
            
            chunk_metadata = {
                **metadata,
                "chunk_type": "operation",
                "operation_id": op_id,
                "endpoint_count": len(endpoints)
            }
            
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _format_endpoint(self, path: str, method: str, operation: Dict,
                        path_params: List, spec: Dict) -> str:
        """Format endpoint information into readable documentation"""
        content = f"## {method.upper()} {path}\n\n"
        
        # Add summary and description
        if "summary" in operation:
            content += f"**Summary**: {operation['summary']}\n\n"
        
        if "description" in operation:
            content += f"**Description**: {operation['description']}\n\n"
        
        # Add tags
        if "tags" in operation:
            content += f"**Tags**: {', '.join(operation['tags'])}\n\n"
        
        # Add parameters
        all_params = path_params + operation.get("parameters", [])
        if all_params:
            content += "### Parameters\n\n"
            for param in all_params:
                content += self._format_parameter(param)
        
        # Add request body (OpenAPI 3.x)
        if "requestBody" in operation:
            content += "### Request Body\n\n"
            content += self._format_request_body(operation["requestBody"])
        
        # Add responses
        if "responses" in operation:
            content += "### Responses\n\n"
            for status, response in operation["responses"].items():
                content += f"#### {status}"
                if isinstance(response, dict) and "description" in response:
                    content += f" - {response['description']}"
                content += "\n\n"
                
                if isinstance(response, dict) and "content" in response:
                    content += self._format_response_content(response["content"])
        
        # Add security requirements
        if "security" in operation:
            content += "### Security\n\n"
            for security in operation["security"]:
                for scheme, scopes in security.items():
                    content += f"- **{scheme}**"
                    if scopes:
                        content += f" (scopes: {', '.join(scopes)})"
                    content += "\n"
            content += "\n"
        
        return content
    
    def _format_parameter(self, param: Dict) -> str:
        """Format a parameter definition"""
        if "$ref" in param:
            return f"- Reference: {param['$ref']}\n"
        
        content = f"- **{param.get('name', 'unnamed')}**"
        content += f" ({param.get('in', 'unknown')})"
        
        if param.get("required", False):
            content += " *[required]*"
        
        content += f": {param.get('description', 'No description')}\n"
        
        if "schema" in param:
            schema = param["schema"]
            content += f"  - Type: `{schema.get('type', 'any')}`"
            if "format" in schema:
                content += f" (format: {schema['format']})"
            if "enum" in schema:
                content += f"\n  - Enum: {', '.join(map(str, schema['enum']))}"
            if "default" in schema:
                content += f"\n  - Default: `{schema['default']}`"
            content += "\n"
        
        return content
    
    def _format_request_body(self, request_body: Dict) -> str:
        """Format request body information"""
        content = ""
        
        if "description" in request_body:
            content += f"{request_body['description']}\n\n"
        
        if request_body.get("required", False):
            content += "*Required*\n\n"
        
        if "content" in request_body:
            for content_type, media_type in request_body["content"].items():
                content += f"**Content-Type**: `{content_type}`\n\n"
                
                if "schema" in media_type:
                    content += "```json\n"
                    content += json.dumps(media_type["schema"], indent=2)
                    content += "\n```\n\n"
                
                if "example" in media_type:
                    content += "**Example**:\n```json\n"
                    content += json.dumps(media_type["example"], indent=2)
                    content += "\n```\n\n"
        
        return content
    
    def _format_response_content(self, content: Dict) -> str:
        """Format response content information"""
        output = ""
        
        for content_type, media_type in content.items():
            output += f"**Content-Type**: `{content_type}`\n\n"
            
            if "schema" in media_type:
                output += "```json\n"
                output += json.dumps(media_type["schema"], indent=2)
                output += "\n```\n\n"
            
            if "example" in media_type:
                output += "**Example**:\n```json\n"
                output += json.dumps(media_type["example"], indent=2)
                output += "\n```\n\n"
        
        return output
    
    def _find_schema_usage(self, schema_name: str, spec: Dict) -> List[str]:
        """Find endpoints that use a specific schema"""
        usage = []
        paths = spec.get("paths", {})
        
        # Search through all operations
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
                
            for method, operation in path_item.items():
                if method not in ["get", "post", "put", "delete", "patch", "options", "head", "trace"]:
                    continue
                
                # Check in request body
                if "requestBody" in operation:
                    if self._contains_schema_ref(operation["requestBody"], schema_name):
                        usage.append(f"{method.upper()} {path} (request)")
                
                # Check in responses
                if "responses" in operation:
                    for status, response in operation["responses"].items():
                        if isinstance(response, dict) and self._contains_schema_ref(response, schema_name):
                            usage.append(f"{method.upper()} {path} (response {status})")
        
        return usage
    
    def _contains_schema_ref(self, obj: Any, schema_name: str) -> bool:
        """Recursively check if an object contains a reference to a schema"""
        if isinstance(obj, dict):
            if "$ref" in obj and schema_name in obj["$ref"]:
                return True
            return any(self._contains_schema_ref(v, schema_name) for v in obj.values())
        elif isinstance(obj, list):
            return any(self._contains_schema_ref(item, schema_name) for item in obj)
        return False
    
    def _create_overview_chunk(self, spec: Dict, metadata: Dict) -> Optional[Dict]:
        """Create an overview chunk for the API"""
        content = f"# API: {metadata['title']}\n\n"
        content += f"**Version**: {metadata['api_version']}\n"
        content += f"**OpenAPI**: {metadata['openapi_version']}\n\n"
        
        if metadata['description']:
            content += f"## Description\n\n{metadata['description']}\n\n"
        
        # Add servers/host information
        if "servers" in spec:
            content += "## Servers\n\n"
            for server in spec["servers"]:
                content += f"- {server.get('url', 'Unknown URL')}"
                if "description" in server:
                    content += f": {server['description']}"
                content += "\n"
            content += "\n"
        elif "host" in spec:  # OpenAPI 2.x
            content += f"## Host\n\n{spec['host']}\n\n"
        
        # Add contact information
        if "contact" in spec.get("info", {}):
            contact = spec["info"]["contact"]
            content += "## Contact\n\n"
            if "name" in contact:
                content += f"- Name: {contact['name']}\n"
            if "email" in contact:
                content += f"- Email: {contact['email']}\n"
            if "url" in contact:
                content += f"- URL: {contact['url']}\n"
            content += "\n"
        
        # Add license information
        if "license" in spec.get("info", {}):
            license_info = spec["info"]["license"]
            content += f"## License\n\n{license_info.get('name', 'Unknown')}"
            if "url" in license_info:
                content += f" - {license_info['url']}"
            content += "\n\n"
        
        # Add tags summary
        if "tags" in spec:
            content += "## Tags\n\n"
            for tag in spec["tags"]:
                content += f"- **{tag.get('name', 'Unknown')}**"
                if "description" in tag:
                    content += f": {tag['description']}"
                content += "\n"
            content += "\n"
        
        # Add endpoints summary
        paths = spec.get("paths", {})
        if paths:
            content += f"## Endpoints Summary\n\n"
            content += f"Total endpoints: {sum(1 for p in paths.values() if isinstance(p, dict) for m in p if m in ['get', 'post', 'put', 'delete', 'patch'])}\n\n"
            
            # Group by tags
            by_tag = {}
            for path, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                for method, operation in path_item.items():
                    if method not in ["get", "post", "put", "delete", "patch", "options", "head", "trace"]:
                        continue
                    tags = operation.get("tags", ["Untagged"]) if isinstance(operation, dict) else ["Untagged"]
                    for tag in tags:
                        if tag not in by_tag:
                            by_tag[tag] = []
                        by_tag[tag].append(f"{method.upper()} {path}")
            
            for tag, endpoints in sorted(by_tag.items()):
                content += f"### {tag} ({len(endpoints)} endpoints)\n"
                for endpoint in endpoints[:5]:  # Show first 5
                    content += f"- {endpoint}\n"
                if len(endpoints) > 5:
                    content += f"- ... and {len(endpoints) - 5} more\n"
                content += "\n"
        
        # Generate chunk ID
        chunk_id = hashlib.md5(f"overview_{metadata['title']}".encode()).hexdigest()
        
        return {
            "id": chunk_id,
            "content": content,
            "metadata": {
                **metadata,
                "chunk_type": "overview"
            }
        }
    
    def _chunk_security_schemes(self, spec: Dict, metadata: Dict) -> List[Dict]:
        """Create chunks for security schemes"""
        chunks = []
        
        # Handle both OpenAPI 3.x and 2.x locations
        security_schemes = {}
        if "components" in spec and "securitySchemes" in spec["components"]:
            security_schemes = spec["components"]["securitySchemes"]
        elif "securityDefinitions" in spec:
            security_schemes = spec["securityDefinitions"]
        
        if not security_schemes:
            return chunks
        
        content = "# Security Schemes\n\n"
        
        for scheme_name, scheme_def in security_schemes.items():
            content += f"## {scheme_name}\n\n"
            
            if "type" in scheme_def:
                content += f"**Type**: {scheme_def['type']}\n\n"
            
            if "description" in scheme_def:
                content += f"{scheme_def['description']}\n\n"
            
            # Handle different security types
            if scheme_def.get("type") == "apiKey":
                content += f"- **In**: {scheme_def.get('in', 'unknown')}\n"
                content += f"- **Name**: {scheme_def.get('name', 'unknown')}\n\n"
            
            elif scheme_def.get("type") == "http":
                content += f"- **Scheme**: {scheme_def.get('scheme', 'unknown')}\n"
                if "bearerFormat" in scheme_def:
                    content += f"- **Bearer Format**: {scheme_def['bearerFormat']}\n"
                content += "\n"
            
            elif scheme_def.get("type") == "oauth2":
                content += "### OAuth2 Flows\n\n"
                if "flows" in scheme_def:
                    for flow_type, flow_def in scheme_def["flows"].items():
                        content += f"#### {flow_type}\n"
                        if "authorizationUrl" in flow_def:
                            content += f"- Authorization URL: {flow_def['authorizationUrl']}\n"
                        if "tokenUrl" in flow_def:
                            content += f"- Token URL: {flow_def['tokenUrl']}\n"
                        if "scopes" in flow_def:
                            content += "- Scopes:\n"
                            for scope, desc in flow_def["scopes"].items():
                                content += f"  - `{scope}`: {desc}\n"
                        content += "\n"
        
        # Add global security requirements
        if "security" in spec:
            content += "## Global Security Requirements\n\n"
            for security in spec["security"]:
                for scheme, scopes in security.items():
                    content += f"- **{scheme}**"
                    if scopes:
                        content += f" (scopes: {', '.join(scopes)})"
                    content += "\n"
        
        chunk_id = hashlib.md5(f"security_{metadata['title']}".encode()).hexdigest()
        
        chunks.append({
            "id": chunk_id,
            "content": content,
            "metadata": {
                **metadata,
                "chunk_type": "security"
            }
        })
        
        return chunks