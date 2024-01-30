class SVMWrapper:
    def __init__(self, model, svm=None):
        super().__init__()
 
        self.model = model
        self.svm = svm
        self.intermediate_model = None
        
        if svm is None:
            self.svm = SVC(kernel='rbf')
 
    def split_layer_capturer(self):
     
        if len(self.model.layers) <= 2:
            raise ValueError('The number of layers is too small to split, more layers are needed')
 
        for layer in self.model.layers:
            if layer.name == "layer_to_split":
                return layer
 
        return self.model.layers[-2]
 
    def fit(self, x=None, y1=None, y=None, batch_size=None, epochs=0, verbose=2, callbacks=None, validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, **kwargs):
        
        fit = self.model.fit(x, y1, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
        
        self.intermediate_model=Model(inputs=self.model.input, outputs=self.split_layer_capturer().output)
        
        intermediate_output = self.intermediate_model.predict(x)
        self.svm.fit(intermediate_output, y)
        
        return fit
 
    def predict(self, x, batch_size=None, verbose=2, steps=None):
       
        intermediate_prediction = self.intermediate_model.predict(x, batch_size, verbose, steps)
        output = self.svm.predict(intermediate_prediction)
 
        return output
 
    def svm_evaluate(self, x=None, y=None, batch_size=None, verbose=2, steps=None):
       
        if self.intermediate_model is None:
            raise Exception("Please fit the model before evaluation")
            
        outputs = self.predict(x, batch_size, verbose, steps)
        is_correct = [outputs[i] == y[i]
                   for i in range(len(outputs))]
 
        accuracy = sum(is_correct) / len(is_correct)
 
        return accuracy
