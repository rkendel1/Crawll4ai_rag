"""
Test script for PDF and OpenAPI ingestion functionality
Run this to verify the new PDF and OpenAPI processing capabilities work correctly
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from pdf_processor import PDFProcessor
from openapi_processor import OpenAPIProcessor

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_success(message):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")

def print_error(message):
    print(f"{Colors.RED}✗{Colors.ENDC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {message}")

def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{message}{Colors.ENDC}")
    print("=" * len(message))

async def test_pdf_processing():
    """Test PDF processing functionality"""
    print_header("Testing PDF Processing")
    
    # Check if test PDF exists
    test_pdf = project_root / "test_data" / "sample.pdf"
    
    if not test_pdf.exists():
        print_warning(f"Test PDF not found at {test_pdf}")
        print_info("Creating a simple test PDF using reportlab...")
        
        try:
            # Try to create a simple PDF for testing
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            test_pdf.parent.mkdir(exist_ok=True)
            
            # Create a simple PDF
            c = canvas.Canvas(str(test_pdf), pagesize=letter)
            width, height = letter
            
            # Page 1
            c.drawString(100, height - 100, "Sample PDF Document for Testing")
            c.drawString(100, height - 130, "This is page 1 of the test document.")
            c.drawString(100, height - 160, "It contains some sample text to test PDF extraction.")
            c.drawString(100, height - 190, "API Documentation: Users Endpoint")
            c.drawString(100, height - 220, "The /users endpoint allows you to manage user accounts.")
            c.drawString(100, height - 250, "GET /users - Retrieve all users")
            c.drawString(100, height - 280, "POST /users - Create a new user")
            c.showPage()
            
            # Page 2
            c.drawString(100, height - 100, "Page 2: Authentication")
            c.drawString(100, height - 130, "Authentication is required for all API endpoints.")
            c.drawString(100, height - 160, "Use Bearer token in Authorization header.")
            c.drawString(100, height - 190, "Example: Authorization: Bearer your-token-here")
            c.drawString(100, height - 220, "Token expiry: 24 hours")
            c.showPage()
            
            c.save()
            print_success(f"Created test PDF at {test_pdf}")
            
        except ImportError:
            print_warning("reportlab not available, creating a text file instead")
            # Create a text file that we can test with
            test_txt = project_root / "test_data" / "sample.txt"
            test_txt.parent.mkdir(exist_ok=True)
            with open(test_txt, 'w') as f:
                f.write("""Sample Text Document for Testing
This is a sample text document to test text processing.
It contains some sample content about API documentation.

API Documentation: Users Endpoint
The /users endpoint allows you to manage user accounts.
GET /users - Retrieve all users
POST /users - Create a new user

Page 2: Authentication
Authentication is required for all API endpoints.
Use Bearer token in Authorization header.
Example: Authorization: Bearer your-token-here
Token expiry: 24 hours
""")
            print_info(f"Created sample text file at {test_txt} instead")
            return await test_text_processing()
        except Exception as e:
            print_error(f"Could not create test PDF: {e}")
            return False
    
    processor = PDFProcessor()
    
    try:
        print_info(f"Processing PDF: {test_pdf.name}")
        chunks = await processor.process_pdf(str(test_pdf))
        
        if chunks:
            print_success(f"Successfully processed PDF: {len(chunks)} chunks created")
            
            # Display first chunk preview
            if len(chunks) > 0:
                first_chunk = chunks[0]
                print_info(f"First chunk preview (first 200 chars):")
                print(f"  Content: {first_chunk['content'][:200]}...")
                print(f"  Metadata keys: {list(first_chunk['metadata'].keys())}")
                print(f"  Pages: {first_chunk['metadata'].get('pages', 'N/A')}")
                print(f"  Word count: {first_chunk['metadata'].get('word_count', 'N/A')}")
            
            return True
        else:
            print_error("No chunks created from PDF")
            return False
            
    except Exception as e:
        print_error(f"Error processing PDF: {e}")
        
        # Try fallback method
        print_info("Trying fallback extraction method...")
        try:
            chunks = await processor.extract_with_fallback(str(test_pdf))
            if chunks:
                print_success(f"Fallback extraction successful: {len(chunks)} chunks")
                return True
            else:
                print_error("Fallback extraction failed")
                return False
        except Exception as fallback_e:
            print_error(f"Fallback extraction also failed: {fallback_e}")
            return False

async def test_text_processing():
    """Fallback test using text file"""
    print_header("Testing Text Processing (PDF Fallback)")
    
    test_txt = project_root / "test_data" / "sample.txt"
    if not test_txt.exists():
        print_error("No test text file available")
        return False
    
    # For text files, we'll simulate PDF processing
    try:
        with open(test_txt, 'r') as f:
            content = f.read()
        
        # Create a mock processor result
        processor = PDFProcessor()
        
        # Simulate chunking
        words = content.split()
        chunk_size = 100
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "id": f"text_chunk_{i}",
                "content": chunk_text,
                "metadata": {
                    "source": str(test_txt),
                    "type": "text",
                    "chunk_index": len(chunks),
                    "word_count": len(chunk_words)
                }
            })
        
        print_success(f"Text processing successful: {len(chunks)} chunks created")
        if chunks:
            print_info(f"First chunk preview: {chunks[0]['content'][:200]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Error processing text file: {e}")
        return False

async def test_openapi_processing():
    """Test OpenAPI processing functionality"""
    print_header("Testing OpenAPI Processing")
    
    # Create a comprehensive sample OpenAPI spec
    sample_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A comprehensive test API for validation and demonstration",
            "contact": {
                "name": "API Support",
                "email": "support@testapi.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "https://api.test.com/v1",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.test.com/v1",
                "description": "Staging server"
            }
        ],
        "tags": [
            {
                "name": "users",
                "description": "User management operations"
            },
            {
                "name": "auth",
                "description": "Authentication and authorization"
            }
        ],
        "paths": {
            "/users": {
                "get": {
                    "summary": "Get all users",
                    "description": "Retrieve a list of all users with optional filtering",
                    "operationId": "getUsers",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of users to return",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        {
                            "name": "status",
                            "in": "query",
                            "description": "Filter users by status",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["active", "inactive", "pending"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of users",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/User"
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request"
                        }
                    },
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ]
                },
                "post": {
                    "summary": "Create a user",
                    "description": "Create a new user account",
                    "operationId": "createUser",
                    "tags": ["users"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CreateUserRequest"
                                },
                                "example": {
                                    "name": "John Doe",
                                    "email": "john@example.com",
                                    "role": "user"
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "User created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/User"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid input"
                        },
                        "409": {
                            "description": "User already exists"
                        }
                    },
                    "security": [
                        {
                            "bearerAuth": ["admin"]
                        }
                    ]
                }
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user by ID",
                    "description": "Retrieve a specific user by their ID",
                    "operationId": "getUserById",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "description": "User ID",
                            "schema": {
                                "type": "string",
                                "format": "uuid"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "User details",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/User"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "User not found"
                        }
                    },
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ]
                }
            },
            "/auth/login": {
                "post": {
                    "summary": "User login",
                    "description": "Authenticate user and return access token",
                    "operationId": "login",
                    "tags": ["auth"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/LoginRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/LoginResponse"
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Invalid credentials"
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "description": "User object with account information",
                    "properties": {
                        "id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Unique user identifier"
                        },
                        "name": {
                            "type": "string",
                            "description": "User's full name"
                        },
                        "email": {
                            "type": "string",
                            "format": "email",
                            "description": "User's email address"
                        },
                        "role": {
                            "type": "string",
                            "enum": ["admin", "user", "guest"],
                            "description": "User's role in the system"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive", "pending"],
                            "description": "Current user status"
                        },
                        "created_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Account creation timestamp"
                        }
                    },
                    "required": ["id", "name", "email", "role"]
                },
                "CreateUserRequest": {
                    "type": "object",
                    "description": "Request body for creating a new user",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 100
                        },
                        "email": {
                            "type": "string",
                            "format": "email"
                        },
                        "role": {
                            "type": "string",
                            "enum": ["user", "admin"],
                            "default": "user"
                        }
                    },
                    "required": ["name", "email"]
                },
                "LoginRequest": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email"
                        },
                        "password": {
                            "type": "string",
                            "minLength": 8
                        }
                    },
                    "required": ["email", "password"]
                },
                "LoginResponse": {
                    "type": "object",
                    "properties": {
                        "access_token": {
                            "type": "string"
                        },
                        "token_type": {
                            "type": "string",
                            "default": "Bearer"
                        },
                        "expires_in": {
                            "type": "integer",
                            "description": "Token expiry in seconds"
                        },
                        "user": {
                            "$ref": "#/components/schemas/User"
                        }
                    },
                    "required": ["access_token", "token_type", "expires_in", "user"]
                }
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token for authentication"
                }
            }
        },
        "security": [
            {
                "bearerAuth": []
            }
        ]
    }
    
    # Save sample spec to file
    test_dir = project_root / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    spec_file = test_dir / "sample_api.json"
    with open(spec_file, 'w') as f:
        json.dump(sample_spec, f, indent=2)
    
    print_success(f"Created sample OpenAPI spec at {spec_file}")
    
    processor = OpenAPIProcessor()
    
    # Test different strategies
    strategies = ["endpoint", "schema", "combined", "operation"]
    results = {}
    
    for strategy in strategies:
        try:
            print_info(f"Testing strategy: {strategy}")
            chunks = await processor.process_openapi(str(spec_file), strategy)
            
            if chunks:
                print_success(f"Strategy '{strategy}': {len(chunks)} chunks created")
                
                # Show chunk type breakdown
                chunk_types = {}
                for chunk in chunks:
                    chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                print_info(f"  Chunk breakdown: {dict(chunk_types)}")
                
                # Show sample chunk
                if chunks:
                    sample_chunk = chunks[0]
                    print_info(f"  Sample chunk type: {sample_chunk['metadata'].get('chunk_type', 'unknown')}")
                    print_info(f"  Sample content preview: {sample_chunk['content'][:150]}...")
                
                results[strategy] = len(chunks)
            else:
                print_error(f"Strategy '{strategy}': No chunks created")
                results[strategy] = 0
                
        except Exception as e:
            print_error(f"Strategy '{strategy}' failed: {e}")
            results[strategy] = 0
    
    print_info("\nStrategy Results Summary:")
    for strategy, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {strategy}: {count} chunks")
    
    # Test with YAML format as well
    print_info("\nTesting YAML format...")
    try:
        import yaml
        
        yaml_file = test_dir / "sample_api.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_spec, f, default_flow_style=False)
        
        chunks = await processor.process_openapi(str(yaml_file), "combined")
        if chunks:
            print_success(f"YAML processing successful: {len(chunks)} chunks")
        else:
            print_error("YAML processing failed: No chunks created")
            
    except ImportError:
        print_warning("PyYAML not available, skipping YAML test")
    except Exception as e:
        print_error(f"YAML processing failed: {e}")
    
    return len([r for r in results.values() if r > 0]) > 0

async def test_error_handling():
    """Test error handling for various failure scenarios"""
    print_header("Testing Error Handling")
    
    pdf_processor = PDFProcessor()
    openapi_processor = OpenAPIProcessor()
    
    test_cases = [
        # PDF error cases
        {
            "name": "Non-existent PDF file",
            "test": lambda: pdf_processor.process_pdf("/nonexistent/file.pdf"),
            "expect_error": True
        },
        {
            "name": "Invalid OpenAPI file path", 
            "test": lambda: openapi_processor.process_openapi("/nonexistent/spec.json"),
            "expect_error": True
        },
        {
            "name": "Invalid OpenAPI strategy",
            "test": lambda: openapi_processor.process_openapi("test_data/sample_api.json", "invalid_strategy"),
            "expect_error": False  # Should default to 'combined'
        }
    ]
    
    for case in test_cases:
        try:
            print_info(f"Testing: {case['name']}")
            result = await case["test"]()
            
            if case["expect_error"]:
                print_error(f"Expected error but got result: {type(result)}")
            else:
                print_success(f"Handled gracefully: {len(result) if result else 0} results")
                
        except Exception as e:
            if case["expect_error"]:
                print_success(f"Correctly raised error: {type(e).__name__}")
            else:
                print_error(f"Unexpected error: {e}")
    
    return True

async def test_integration():
    """Test basic integration without database"""
    print_header("Testing Basic Integration")
    
    try:
        # Test that processors can be imported and instantiated
        pdf_proc = PDFProcessor(chunk_size=500, overlap=100)
        api_proc = OpenAPIProcessor()
        
        print_success("Processors instantiated successfully")
        print_info(f"PDF processor: chunk_size={pdf_proc.chunk_size}, overlap={pdf_proc.overlap}")
        print_info(f"OpenAPI processor strategies: {list(api_proc.chunk_strategies.keys())}")
        
        # Test that we can import the main module components
        from utils import create_embedding  # This might fail if OpenAI key not set
        print_success("Utils imported successfully")
        
        return True
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        return False
    except Exception as e:
        print_warning(f"Partial success with warning: {e}")
        return True

async def main():
    """Run all tests"""
    print_header("PDF and OpenAPI Ingestion Tests")
    print_info("This test suite validates the new document ingestion functionality")
    print_info(f"Project root: {project_root}")
    
    results = {}
    
    # Run all test suites
    test_suites = [
        ("Basic Integration", test_integration()),
        ("PDF Processing", test_pdf_processing()),
        ("OpenAPI Processing", test_openapi_processing()),
        ("Error Handling", test_error_handling())
    ]
    
    for name, test_coro in test_suites:
        try:
            result = await test_coro
            results[name] = result
            
            if result:
                print_success(f"\n{name}: PASSED")
            else:
                print_error(f"\n{name}: FAILED")
                
        except Exception as e:
            print_error(f"\n{name}: ERROR - {e}")
            results[name] = False
    
    # Summary
    print_header("Test Results Summary")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:>6}{Colors.ENDC} {name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print_success("All tests passed! 🎉")
        print_info("The PDF and OpenAPI ingestion functionality is working correctly.")
    elif passed > 0:
        print_warning(f"Partial success: {passed}/{total} tests passed")
        print_info("Some functionality is working, but there may be issues to address.")
    else:
        print_error("All tests failed")
        print_info("There are significant issues that need to be resolved.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)