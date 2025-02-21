import graphviz

# Create a directed graph using Graphviz for a cleaner look
dot = graphviz.Digraph(format='png')

# Define steps with labels
steps = [
    "1. Define Application Requirements",
    "2. Choose or Train a Speech Model",
    "3. Convert the Model for Edge Deployment",
    "4. Preprocess Audio Input",
    "5. Develop Microcontroller Code",
    "6. Optimize for Performance",
    "7. Deploy to the Microcontroller",
    "8. Test and Debug",
    "9. Fine-Tune and Iterate"
]

# Add nodes to the graph
for step in steps:
    dot.node(step, step, shape="box", style="filled", fillcolor="lightblue")

# Add directional edges
for i in range(len(steps) - 1):
    dot.edge(steps[i], steps[i + 1])

# Render and display the diagram
dot.render('speech_model_deployment_graph', view=True)