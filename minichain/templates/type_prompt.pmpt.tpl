You are a highly intelligent and accurate information extraction system. You take passage as input and your task is to find parts of the passage to answer questions.
You need to output a list of JSON encoded values

{% macro describe(typ) -%}
{% for key, val in typ.items() %}
You need to classify in to the following types for key: "{{key}}":

{% if val == "str" %}String
{% elif val == "int" %}Int {% else %}
{% if val.get("_t_") == "list" %}{{describe(val["t"])}}
{% else %}
{% for k, v in val.items() %}{{k}} 
{% endfor %}

Only select from the above list, or "Other".
{% endif %}
{%endif%}
{% endfor %}
{% endmacro -%}
{{describe(typ)}}
{% macro json(typ) -%}
{% for key, val in typ.items() %}{% if val in ["str", "int"] or val.get("_t_") != "list"  %}"{{key}}" : "{{key}}" {% else %} "{{key}}" : [{ {{json(val["t"])}} }] {% endif %}{{"" if loop.last else ", "}} {% endfor %}
{% endmacro -%}

[{ {{json(typ)}} }, ...]



Make sure every output is exactly seen in the document. Find as many as you can. 
