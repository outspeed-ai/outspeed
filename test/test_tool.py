from enum import Enum
from typing import List, Union
from pydantic import BaseModel

import pytest

from outspeed.tool import Tool


class Table(str, Enum):
    orders = "orders"
    customers = "customers"
    products = "products"


class Column(str, Enum):
    id = "id"
    status = "status"
    expected_delivery_date = "expected_delivery_date"
    delivered_at = "delivered_at"
    shipped_at = "shipped_at"
    ordered_at = "ordered_at"
    canceled_at = "canceled_at"


class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="


class OrderBy(str, Enum):
    asc = "asc"
    desc = "desc"


class DynamicValue(BaseModel):
    column_name: str


class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]


class Query(BaseModel):
    table_name: Table
    columns: List[Column]
    conditions: List[Condition]
    order_by: OrderBy


class MockTool(Tool):
    async def run(self, input_parameters: BaseModel):
        # Mock implementation for testing
        return input_parameters


@pytest.fixture
def sample_query():
    return Query(
        table_name=Table.orders,
        columns=[Column.id, Column.status, Column.expected_delivery_date],
        conditions=[
            Condition(
                column=Column.status,
                operator=Operator.eq,
                value="shipped",
            ),
            Condition(
                column=Column.expected_delivery_date,
                operator=Operator.lt,
                value="2023-10-01",
            ),
        ],
        order_by=OrderBy.asc,
    )


@pytest.mark.asyncio
async def test_query_to_openai_tool_json_and_back(sample_query):
    # Initialize the mock tool with Query as parameters_type and Query as response_type
    tool = MockTool(
        name="query_tool",
        description="Converts Query to OpenAI tool JSON and vice versa",
        parameters_type=Query,
        response_type=Query,
    )

    # Convert Query to OpenAI tool call JSON
    tool_json = tool.to_openai_tool_json()

    assert tool_json == {
        "type": "function",
        "function": {
            "name": "query_tool",
            "strict": True,
            "parameters": {
                "$defs": {
                    "Column": {
                        "enum": [
                            "id",
                            "status",
                            "expected_delivery_date",
                            "delivered_at",
                            "shipped_at",
                            "ordered_at",
                            "canceled_at",
                        ],
                        "title": "Column",
                        "type": "string",
                    },
                    "Condition": {
                        "properties": {
                            "column": {"title": "Column", "type": "string"},
                            "operator": {"$ref": "#/$defs/Operator"},
                            "value": {
                                "anyOf": [{"type": "string"}, {"type": "integer"}, {"$ref": "#/$defs/DynamicValue"}],
                                "title": "Value",
                            },
                        },
                        "required": ["column", "operator", "value"],
                        "title": "Condition",
                        "type": "object",
                        "additionalProperties": False,
                    },
                    "DynamicValue": {
                        "properties": {"column_name": {"title": "Column Name", "type": "string"}},
                        "required": ["column_name"],
                        "title": "DynamicValue",
                        "type": "object",
                        "additionalProperties": False,
                    },
                    "Operator": {"enum": ["=", ">", "<", "<=", ">=", "!="], "title": "Operator", "type": "string"},
                    "OrderBy": {"enum": ["asc", "desc"], "title": "OrderBy", "type": "string"},
                    "Table": {"enum": ["orders", "customers", "products"], "title": "Table", "type": "string"},
                },
                "properties": {
                    "table_name": {"$ref": "#/$defs/Table"},
                    "columns": {"items": {"$ref": "#/$defs/Column"}, "title": "Columns", "type": "array"},
                    "conditions": {"items": {"$ref": "#/$defs/Condition"}, "title": "Conditions", "type": "array"},
                    "order_by": {"$ref": "#/$defs/OrderBy"},
                },
                "required": ["table_name", "columns", "conditions", "order_by"],
                "title": "Query",
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Converts Query to OpenAI tool JSON and vice versa",
        },
    }

    print(sample_query.model_dump_json())
    # Simulate receiving a tool call
    function_json = {
        "id": "func_123",
        "function": {
            "name": tool.name,
            "arguments": sample_query.model_dump(),
        },
    }

    # Run the tool with the function JSON
    response_json = await tool._run_tool(function_json)

    # Extract the content and convert back to Query
    response_content = response_json["content"]
    reconstructed_query = Query.model_validate_json(response_content)

    # Assert that the original query and the reconstructed query are the same
    assert sample_query == reconstructed_query
