model_desc = json.loads(model_description)

# Building the model
model = keras.Sequential()
input_layer = model_desc['input_layer']
model.add(keras.layers.InputLayer(input_shape=input_layer['shape']))

for layer in model_desc['hidden_layers']:
    if layer['layer_type'] == "Dense":
        model.add(keras.layers.Dense(layer['units'], 
                                     activation=layer['activation'], 
                                     kernel_regularizer=l2() if 'regularization' in layer and layer['regularization'] == 'l2' else None))
    elif layer['layer_type'] == "Dropout":
        model.add(keras.layers.Dropout(layer['rate']))

output_layer = model_desc['output_layer']
model.add(keras.layers.Dense(output_layer['units'], activation=output_layer['activation']))

# Compiling the model
optimizer_desc = model_desc['optimizer']
if optimizer_desc['type'] == "Adam":
    optimizer = keras.optimizers.Adam(learning_rate=optimizer_desc['learning_rate'], decay=optimizer_desc['decay'])

model.compile(optimizer=optimizer,
              loss=model_desc['loss'],
              metrics=model_desc['metrics'])

# Printing the summary
model.summary()
