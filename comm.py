import math

import copy

import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action
# import torch_geometric
'''
Dynamic graph version comm
'''

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.continuous = args.continuous
        self.ngrids = args.obstacles

        self.graph_f = 0
        #self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available else
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])

        '''
        ### finised running in 7.28 best succ 0.85
        self.mapdecode = nn.Sequential(
            nn.Linear(self.args.hid_size, self.args.hid_size),
            nn.Linear(self.args.hid_size, 6*11),
            nn.LeakyReLU()
        )

        self.gnn_decoder = nn.Sequential(
                nn.Linear(self.args.hid_size, self.args.hid_size),
                nn.Linear(self.args.hid_size, 6*6),
                nn.LeakyReLU()
                )
        
        self.griddecode = nn.Sequential(
            nn.Linear(args.hid_size, (self.args.dim - 2 * 3 + 2) * (self.args.dim - 2 * 3 + 2) * 4),
            nn.Unflatten(1, torch.Size([4, (self.args.dim - 2 * 3 + 2), (self.args.dim - 2 * 3 + 2)])),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(4, 16, 3),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 1, 3),
        )
        '''
        self.mapdecode = nn.Sequential(
            nn.Linear(self.args.hid_size, self.args.hid_size),
            nn.Linear(self.args.hid_size, (self.args.nagents+1) * (2+2)), # +9    +2  +self.args.nagents+1+1    +self.args.nagents+1+2
            nn.LeakyReLU()
        )
        self.init_dim = 5  # Initial feature map size
        # self.cnndecode = nn.Sequential(
        #     nn.Linear(args.hid_size*2, 32 * self.init_dim * self.init_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Unflatten(1, (32, self.init_dim, self.init_dim)),  # → (batch, 64, 5, 5)
        #
        #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (batch, 32, 10, 10)
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),  # → (batch, 16, 10, 10)
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(8, 3, kernel_size=1),  # → (batch, 3, 10, 10)
        #     nn.ReLU(inplace=True)   # was nn.Sigmoid
        # )

        # my multi-head attention model
        # --- tarmac model ----
        self.state2query = nn.Linear(args.hid_size, 16)
        self.state2key = nn.Linear(args.hid_size, 16)
        self.state2value = nn.Linear(args.hid_size, 1*args.hid_size)
        self.W_O = nn.Linear(1 * args.hid_size, args.hid_size, bias=False)

        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        '''
        ### finised running in 7.28 best succ 0.85  gcn_ppnode_obs_dr08
        self.gcn_encoder = gcn_mod.GCN(2+self.args.nagents+1+1+9, 32, 16, 0.8)  # pp 9 16 16, grf 8 16 16, pp input with obs 17 16 16
        self.encoder = nn.Sequential(
                nn.Linear(16, args.hid_size),
                nn.ReLU(),
                nn.Linear(args.hid_size, args.hid_size),
                )
        
        self.gcn_encoder = gcn_mod.GAT(2+self.args.nagents+1+1+9, 32, 16, 0.1, 0.2, 3)
        self.encoder = nn.Sequential(
            nn.Linear( 16, args.hid_size),
            nn.ReLU(),
            nn.Linear(args.hid_size, args.hid_size),
        )'''
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(self.hid_size, args.hid_size)
            # self.f_module = nn.RNNCell(self.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(self.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(self.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)
        '''
        self.aggregator = nn.Sequential(
            nn.Linear(args.hid_size, args.hid_size*2),
            nn.LeakyReLU(),
            nn.Linear(args.hid_size*2, args.hid_size),
            nn.LeakyReLU(),
            nn.Linear(args.hid_size, args.hid_size),
            nn.LeakyReLU())
        '''
        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.args.hid_size, 1)
        self.value_global = nn.Linear(self.args.hid_size, self.args.nagents)
        # self.value_global = nn.Linear(self.args.hid_size, 1) # for x+msg


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n
        #agent_mask = agent_mask.to(self.device)
        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask
    '''
    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, adj, extras = x
            # x = torch.squeeze(x)
            node_fmap = []
            for i in range(len(x)):
                ck = self.gcn_encoder(x[i][0], adj[i])
                node_fmap.append(ck[x[i][1]])
            x = torch.stack(node_fmap)
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state
    '''
    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None
        
        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state

    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        batch_size = 1
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            #comm_action = torch.ones(self.nagents)
            #comm_action = comm_action.to(self.device)
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            # agent_mask *= comm_action_mask.double()
            agent_mask = agent_mask * comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.args.hid_size) if self.args.recurrent else hidden_state
            # comm = x.view(batch_size, n, self.args.hid_size) if self.args.recurrent else hidden_state # communicate observation
            masked_msg = torch.zeros(comm.shape) * -1e9
            for idx, agt in enumerate(comm_action):
                if agt == 1:
                    masked_msg[0, idx, :] = comm[0, idx, :]
            query = self.state2query(comm).view(1, self.nagents, -1)
            key = self.state2key(masked_msg).view(1, self.nagents, -1)
            value = self.state2value(masked_msg).view(1, self.nagents, -1)

            # scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hid_size)

            attn = F.softmax(scores, dim=-1)
            comm = torch.matmul(attn, value)

            context = comm.transpose(0, 1).contiguous().view(self.nagents,
                                                             1 * self.args.hid_size)
            # comm = self.W_O(context)

            c = self.C_modules[i](context).unsqueeze(0) #squeeze()

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c 

                #inp = torch.cat((x, c), dim=-1)
                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))
                # output = self.f_module(inp, hidden_state)

                hidden_state = output[0]
                cell_state = output[1]

                # decoded = self.mapdecode(hidden_state)  # for map decoder original
                node = self.mapdecode(hidden_state)
                decoder_input = torch.cat((hidden_state, cell_state), 1)
                # cnn = self.cnndecode(decoder_input)


            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)
                node = self.mapdecode(hidden_state)
                
                # cnn = self.cnndecode(hidden_state)
                

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        '''
        vg = []
        for i in range(len(x)):
            # agent_hs = his_knowledge[i].repeat(self.args.nagents,1)
            agent_hs = x[i].repeat(self.args.nagents,1)
            single_value_global = self.value_global(agent_hs + masked_msg.squeeze()).sum(0)# .view(value_head.shape[0], 1)
            vg.append(single_value_global)
        value_global = torch.stack(vg)
        '''
        value_global = self.value_global(hidden_state).view(value_head.shape[0],value_head.shape[0])#1
        h = hidden_state.view(batch_size, n, self.args.hid_size) #origin
        # act_inp = torch.cat((hidden_state, c), dim=-1)         # cat inp
        # h = act_inp.view(batch_size, n, self.args.hid_size*2)


        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actionss
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
        # action[-1] = action[-1].detach()
        if self.args.recurrent:
            return action, value_head, value_global, (hidden_state.clone(), cell_state.clone()), node#,, cnn grid , auto_res
        else:
            return action, value_head, value_global, node #,, cnn grid

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.args.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.args.hid_size, requires_grad=True)))



