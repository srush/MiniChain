------
toc_depth: 3
------

# API

![image](https://user-images.githubusercontent.com/35882/221280012-d58c186d-4da2-4cb6-96af-4c4d9069943f.png)

::: minichain.Prompt
    options:
        members: [__init__, prompt, parse, __call__]
        

## Chain

::: minichain.Prompt.chain
    options:
        heading_level: 3

![image](https://user-images.githubusercontent.com/35882/221281771-3770be96-02ce-4866-a6f8-c458c9a11c6f.png)

::: minichain.Prompt.map
    options:
        heading_level: 3

![image](https://user-images.githubusercontent.com/35882/221283494-6f76ee85-3652-4bb3-bc42-4e961acd1477.png)



## Special Prompts

::: minichain.SimplePrompt
    options:
        members: false
        heading_level: 3

::: minichain.TemplatePrompt
    options:
        members: [template_file]
        heading_level: 3



::: minichain.EmbeddingPrompt
    options:
        members: [find]
        heading_level: 3

![image](https://user-images.githubusercontent.com/35882/221387303-e3dd8456-a0f0-4a70-a1bb-657fe2240862.png)


::: minichain.TypedTemplatePrompt
    options:
        members: [find]
        heading_level: 3

## Backends

::: minichain.MiniChain
    options:
        members: []
        heading_level: 3

::: minichain.OpenAIBase
    options:
        merge_init_into_class: true
        members: false
        heading_level: 3
        
::: minichain.HuggingFaceBase
    options:
        merge_init_into_class: true
        members: false
        heading_level: 3
        
::: minichain.Manifest
    options:
        merge_init_into_class: true
        members: false
        heading_level: 3


## Sessions and Logging

::: minichain.start_chain
    options:
        heading_level: 3


::: minichain.show_log
    options:
        heading_level: 3



