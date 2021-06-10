from torch import nn

import torch.nn.functional as F
from torchvision import models 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Pre-trained model ResNet18
        self.model_conv = models.resnet18(pretrained=True)

        # freeze the gradients
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # TODO MAKE THE CORRECT INPUT SHAPE! 
        self.fc1 = nn.Linear(320, 50)    # input should be: out_channels times image size times image size 
        self.fc2 = nn.Linear(50, 2)

        # Add dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.model_conv(x))

        print(x.shape)

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        wandb.init(config=args)
        

        # TODO: Implement training loop here 

        model = Classifier()

        # Visualization
        wandb.watch(model, log_freq=100)

        #train_set, _ = ???
        #trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        # negative log likelihood because of the log_softmax
        criterion = nn.NLLLoss()

        # stochastic gradient descent --> adaptive SGD instead 
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 10
        
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # TRAINING
                # clear gradients
                optimizer.zero_grad()

                # use model to predict
                output = model(images)

                # calculate loss
                loss = criterion(output, labels)

                # backpropagate
                loss.backward()

                # take a gradient step/ optimize the weights to min loss
                optimizer.step()
                running_loss += loss.item()

            else:  

                # Find the epoch and corresponding training loss for plotting
                wandb.log({"loss": running_loss/len(trainloader)})
                wandb.log({"batch images for epoch "+str(e) : [wandb.Image(i) for i in images]})
                
                print(f"Training loss: {running_loss/len(trainloader)}")
                
        torch.save(model.state_dict(), 'ResNet18.pth')

def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            checkpoint = torch.load('ResNet18.pth')
        _, test_set = mnist()
        model = Classifier()
        model.load_state_dict(checkpoint)

        # TODO load the test data
        #testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

        running_accuracy = []
        with torch.no_grad():
            # set model to evaluation mode
            model.eval()

            # validation pass here
            for images, labels in testloader:

                ps = torch.exp(model(images))

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(top_class.shape)

                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy.append(accuracy)


        print(f'Accuracy: {np.mean(running_accuracy)*100}%')