import numpy as np

data = open('kafka.txt', 'r').read()

chars = (list(set(data)))
data_size, vocab_size = len(data), len(chars)
# print(data_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
# print(char_to_ix)
# print(ix_to_char)

vector_for_char_a = np.zeros((vocab_size,1))
vector_for_char_a[char_to_ix['a']] = 1
print(vector_for_char_a.ravel())

'''Hyperparameters'''
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

'''Model Parameters'''
wxh = np.random.randn(hidden_size, vocab_size) * 0.01
whh = np.random.randn(hidden_size, hidden_size) * 0.01
why = np.random.randn(hidden_size, vocab_size) * 0.01
bias_h = np.zeros((hidden_size, 1))
bias_y = np.zeros((vocab_size, 1))

'''Loss Function'''
def loss_func(input, target, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    
    loss = 0

    '''forward pass'''
    for t in xrange(len(input)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][input[t]] = 1
        hs[t] = np.tanh(np.dot(wxh,xs[t])+np.dot(whh, hs[t-1])+bias_h)
        ys[t] = np.dot(why, hs[t]) + bias_y
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][target[t], 0])

    dwxh, dwhh, dwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
    dbias_h, dbias_y = np.zeros_like(bias_h), np.zeros_like(bias_y)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(input))):
        dy = np.copy(ps[t])
        dy[target[t]] -= 1
        dwhy += np.dot(dy, hs[t].T)
        dbias_y += dy
        dh = np.dot(why.T, dy) + dh_next
        dhraw = (1-hs[t]*hs[t]) * dh
        dbias_h += dhraw
        dwxh += np.dot(dhraw, xs[t].T)
        dwhh += np.dot(dhraw, hs[t-1].T)
        dh_next = np.dot(whh.T, dhraw)

    for dparam in [dwxh, dwhh, dwhy, dbias_h, dbias_y]:
        np.clip(dparam, -5, 5, out=dparam)
    
    return loss, dwxh, dwhh, dwhy, dbias_h, dbias_y, hs[len(inputs)-1]


'''Prediction (One Full Forward Pass)'''
def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(wxh, x) + np.dot(whh, h) + bias_h)
        y = np.dot(why, h) + bias_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print("\n\n",txt,"\n\n")

hprev = np.zeros((hidden_size,1)) # reset RNN memory  

sample(hprev,char_to_ix['a'],200)

p=0  
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
print "inputs", inputs
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
print "targets", targets

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
mbh, mby = np.zeros_like(bias_h), np.zeros_like(bias_y)
smooth_loss = -np.log(1.0/vocab_size)*seq_length
while n<=1000*100:
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1))
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 1000 == 0:
        print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
        sample(hprev, inputs[0], 200)

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1