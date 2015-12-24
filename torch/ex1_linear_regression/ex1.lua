require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'csvigo'

data = torch.Tensor{ csvigo.load{path='ex1data1.txt', mode='raw'} }
data = data[1]

model = nn.Sequential()
ninputs = 1; noutputs = 1
model:add(nn.Linear(ninputs, noutputs))

criterion = nn.MSECriterion()
x, dl_dx = model:getParameters()

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end
   local sample = data[_nidx_]
   local target = sample[{ {1} }]      -- this funny looking syntax allows
   local inputs = sample[{ {2} }]    -- slicing of arrays.
   dl_dx:zero()
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
   return loss_x, dl_dx
end


sgd_params = {
   learningRate = 1e-4,
   learningRateDecay = 1e-5,
   weightDecay = 0,
   momentum = 0
}

for i = 1,1e4 do
   current_loss = 0
   for i = 1,(#data)[1] do
      _,fs = optim.sgd(feval,x,sgd_params)
      current_loss = current_loss + fs[1]
   end
   current_loss = current_loss / (#data)[1]
   if i%100 == 0 then
        print('current loss = ' .. current_loss)
   end
end


test = torch.Tensor{ csvigo.load{path='ex1data2.txt', mode='raw'} }
test = test[1]

print('id  approx   text')
current_loss = 0
for i = 1,(#test)[1] do
   local myPrediction = model:forward(test[i][1])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], test[i][2]))
   current_loss = current_loss + (myPrediction[1] - test[i][2])*(myPrediction[1] - test[i][2])
end
current_loss = current_loss / (#test)[1]
print('evaluation = ' .. current_loss)
