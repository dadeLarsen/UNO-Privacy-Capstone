## Requirements outline

### This Requirements document will eventually become Chapter 2 in the final report.

It will follow the outline below:

1. Goals
    1. Goal Model - Show relationships between stakeholder goals. Follow i* modeling format (expecting SRMs but can also have SDMs if there are significant dependencies between actors). Initial goal model can be sketched by hand. Final version must use an i* modeling tool such as [[PiStar](http://www.cin.ufpe.br/~jhcp/pistar)]
    2. Goal Descriptions - Brief descriptions of each object in the goal model.
2. Functional Requirements - Specification of services your product should provide.
    - General format: [unique ID] [[MoSCoW](https://en.wikipedia.org/wiki/MoSCoW_method) designation][specification]
    - The specification of a functional requirement is generally written concisely as a user story: As a [stakeholder] so that [goal] I want to [statement of service]
    - In addition to a concise statement of the service or functionality, additional information may be needed to explain it in more nuanced detail. This can take one or more of the following forms as appropriate:
        - a narrative paragraph that explains the user story.
        - BDD scenarios
        - Use case description: use case name, actors, pre/post conditions, flow of events, remarks
            - (use case descriptions can be used as alternative to user stories + BDD scenarios)
            - for use cases, a single use case diagram is expected that includes all use cases
            - the rationale must refer to some goal in the goal model
        - Tabular: specify condition (stimulus) and action (response)
        - Organize the list of requirements logically, that is, group related requirements together.
        - Game-based products can also specify requirements in terms of game elements: theme, story, gameplay, game assets, etc.
3. Wireframes - rough sketches used as a frame of reference in discussing requirements
    - draw by hand or use a wireframing tool like [Balsamiq](https://balsamiq.com/).
    - see [Paper Prototyping](https://mynu.instructure.com/courses/40667/files/6239335/download?    wrap=1) lecture for reference, espcially:
        - make sure sample data shown is consistent across screens
        - use greeking where the data/text is irrelevant to the behavior of application
        - divide and conquer - divide up the wireframe creation activity, but the requirements  analyst should coordinate it so that the prototype screens are consistent with each other
    - include unique ID and description for each sketch
4. Object Model - simple UML class diagram sketch showing relationships between relevant entities from the Subject Facet of your Context
    - each class is an entity from the Subject Facet that needs to be represented in your application
    - the UML class diagram just needs class names; skip the attributes and methods
    - useful as a starting point for defining persistent data requirements
    - include a brief description indicating which entities are required and which ones are optional
5. Nonfunctional Requirements [as needed]
    - General format: [unique ID] [MoSCoW designation] [specification]
    - For the specification itself, use GQM to document analysis requirements and product-based     nonfunctional requirements. Use narrative text to document other types of nonfunctional     requirements.
    - Relation to goals - requirements should be derived from goals in the goal model
6. Analysis Requirements [as needed]
    - same outline as Nonfunctional Requirements
7. External System Interfaces [as needed]
    - API specifications of existing systems that interact with your product, e.g., the name of the     application being accessed, services called, parameters passed, outputs returned, set up    instructions (e.g., API keys)
    - file formats your product is required to use, e.g., if you had to input from, or output to a file with a certain csv format
8. Requirements not Implemented [for final report] - this section lists requirements specified early in the semester but not implemented and the reasons, e.g., lack of time, changes in project direction, etc. 


Some general principles to remember:

1. Avoid redundancy (DRY principle)
2. Every requirement must be related to some goal in the goal model
3. Start with requirements you understand the most and add the rest as you learn more about them
4. Use the unique IDs to cross-reference related requirements as needed