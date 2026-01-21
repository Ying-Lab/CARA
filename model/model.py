import torch
import torch.nn as nn
from model.custom_mlp import MLP
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DistAlignEMA:
    def __init__(self, num_classes, momentum=0.99, p_target_type='model'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.p_target_type = p_target_type
        self.p_model = torch.ones(num_classes) / num_classes
        
    def update(self, probs_lb):
        if probs_lb is not None:
            p_model_batch = probs_lb.mean(dim=0)
            # 确保设备一致性
            self.p_model = self.p_model.to(p_model_batch.device)
            self.p_model = self.momentum * self.p_model + (1 - self.momentum) * p_model_batch
            
    def dist_align(self, probs_ulb):
        if self.p_target_type == 'model':
            p_target = self.p_model
        else:
            p_target = torch.ones(self.num_classes) / self.num_classes
            
        p_target = p_target.to(probs_ulb.device)
        probs_ulb_avg = probs_ulb.mean(dim=0)
        probs_ulb = probs_ulb * (p_target / (probs_ulb_avg + 1e-8)).unsqueeze(0)
        return probs_ulb

class cara(nn.Module):
    def __init__(self,
                 output_size = 10,
                 rna_input_size = 1000,
                 atac_input_size= 1000,
                 z_dim = 50,                 
                 hidden_layers = [500,],
                 batch_num = 2,
                 use_cuda = False,
                 config_enum = None,
                 aux_loss_multiplier = None,
                 alpha=0.01,
                 # 新增类别权重相关参数
                 use_class_weights=False,
                 class_weight_method='balanced',  # 'balanced', 'inverse_freq', 'manual'
                 manual_class_weights=None,
                 confidence_threshold=0.8,  # 伪标签置信度阈值
                 weight_update_frequency=100,  # 权重更新频率
    ):
        super().__init__()

        # 原有参数
        self.output_size = output_size
        self.rna_input_size = rna_input_size
        self.atac_input_size = atac_input_size
        self.z_dim = z_dim 
        self.batch_discriminator_output_size = batch_num
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.epsilon = 1e-8
        self.adversarial_alpha = 0.5
        self.alpha = alpha

        # 新增类别权重相关参数
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method
        self.manual_class_weights = manual_class_weights
        self.confidence_threshold = confidence_threshold
        self.weight_update_frequency = weight_update_frequency
        
        # 权重相关状态
        self.class_weights_tensor = None
        self.labeled_class_counts = torch.zeros(output_size)
        self.unlabeled_class_counts = torch.zeros(output_size)
        self.total_samples_seen = 0
        self.update_counter = 0
        
        # 设置网络
        self.setup_networks()
        self.dist_aligner = DistAlignEMA(self.output_size, 0.99, p_target_type='model')

    def setup_networks(self):
        hidden_sizes = self.hidden_layers
        
        # 原有网络结构保持不变
        self.encoder_rna2z = MLP(
            [self.rna_input_size] + hidden_sizes + [[self.z_dim, self.z_dim]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Sigmoid],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.encoder_z2y = MLP(
            [self.z_dim] + hidden_sizes + [self.output_size],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.encoder_rna_ls = MLP(
            [self.rna_input_size] + hidden_sizes + [[self.rna_input_size, self.rna_input_size]],
            activation=nn.Softplus,
            output_activation=[nn.Softplus, nn.Softplus],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.encoder_atac_ls = MLP(
            [self.atac_input_size] + hidden_sizes + [[self.atac_input_size, self.atac_input_size]],
            activation=nn.Softplus,
            output_activation=[nn.Softplus, nn.Softplus],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.decoder_z2zy = MLP(
            [self.z_dim + self.output_size] + hidden_sizes + [[self.z_dim, self.z_dim]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Sigmoid],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.decoder_rna_thetas = MLP(
            [self.z_dim] + hidden_sizes + [[self.rna_input_size, self.rna_input_size]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Softmax],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.decoder_atac_thetas = MLP(
            [self.z_dim] + hidden_sizes + [[self.atac_input_size, self.atac_input_size]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Softmax],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.encoder_atac2z = MLP(
            [self.atac_input_size] + hidden_sizes + [[self.z_dim, self.z_dim]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Sigmoid],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.batch_discriminator = MLP(
            [self.z_dim] + hidden_sizes + [self.batch_discriminator_output_size],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.encoder_z = MLP(
            [self.z_dim + self.output_size] + hidden_sizes + [[self.z_dim, self.z_dim]],
            activation=nn.Softplus,
            output_activation=[nn.Sigmoid, nn.Sigmoid],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        
        self.rna_zero_inflation_layer = nn.Linear(self.rna_input_size, 1)
        self.atac_zero_inflation_layer = nn.Linear(self.atac_input_size, 1)
        self.cutoff = nn.Threshold(1.0e-9, 1.0e-9)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
        if self.use_cuda:
            self.to(device)

    def compute_initial_class_weights(self, labeled_dataloader):
        if not self.use_class_weights:
            return torch.ones(self.output_size, device=device)

        all_labels = []
        
        for batch_data in labeled_dataloader:
            if len(batch_data) >= 2:
                _, labels = batch_data[0], batch_data[1]
                if labels is not None:
                    if len(labels.shape) > 1:
                        labels = labels.argmax(dim=1)
                    all_labels.extend(labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return torch.ones(self.output_size, device=device)
        
        all_labels = np.array(all_labels)
        
        # 添加调试信息
        unique_labels = np.unique(all_labels)

        if self.class_weight_method == 'balanced':
            # 关键修改：只使用数据中实际存在的类别
            existing_classes = unique_labels
            
            # 使用sklearn计算存在类别的权重
            weights = compute_class_weight('balanced', classes=existing_classes, y=all_labels)
            
            # 创建完整的权重张量，默认权重为1.0
            class_weights = torch.ones(self.output_size, dtype=torch.float32, device=device)
            
            # 为存在的类别设置计算出的权重
            for i, class_idx in enumerate(existing_classes):
                if class_idx < self.output_size:  # 确保索引有效
                    class_weights[class_idx] = weights[i]
            
            
        else:
            # 逆频率权重
            unique, counts = np.unique(all_labels, return_counts=True)
            total_samples = len(all_labels)
            weights = total_samples / (len(unique) * counts)
            
            class_weights = torch.ones(self.output_size, dtype=torch.float32, device=device)
            for i, class_idx in enumerate(unique):
                if class_idx < self.output_size:  # 确保索引有效
                    class_weights[class_idx] = weights[i]
        
        return class_weights


    def update_class_weights_with_pseudolabels(self, predictions, confidences):
        """
        基于伪标签更新类别权重
        """
        if not self.use_class_weights:
            return
        
        self.update_counter += 1
        
        # 只在指定频率更新权重
        if self.update_counter % self.weight_update_frequency != 0:
            return
        
        # 获取高置信度的伪标签
        high_conf_mask = confidences.max(dim=1)[0] > self.confidence_threshold
        if high_conf_mask.sum() == 0:
            return
        
        pseudo_labels = predictions[high_conf_mask].argmax(dim=1)
        
        # 更新无标签数据的类别计数
        for label in pseudo_labels:
            self.unlabeled_class_counts[label.item()] += 1
        
        # 重新计算权重
        total_counts = self.labeled_class_counts + self.unlabeled_class_counts
        
        # 避免除零
        total_counts = torch.clamp(total_counts, min=1.0)
        
        # 计算新的权重
        total_samples = total_counts.sum()
        new_weights = total_samples / (self.output_size * total_counts)
        
        # 平滑更新权重
        if self.class_weights_tensor is not None:
            momentum = 0.9
            self.class_weights_tensor = (momentum * self.class_weights_tensor + 
                                       (1 - momentum) * new_weights.to(device))
        else:
            self.class_weights_tensor = new_weights.to(device)
        
        # 归一化
        self.class_weights_tensor = (self.class_weights_tensor / 
                                   self.class_weights_tensor.sum() * self.output_size)

    def batch_adv(self, z, adversarial_alpha=None):
        z_rev = GradientReversalFn.apply(z, adversarial_alpha)
        return self.batch_discriminator(z_rev)

    def model(self, xs, ys=None, mode=torch.tensor([1., 0.]), batch=None):
        """
        原有的生成模型，保持不变
        """
        pyro.module('scc', self)
        xs = xs.to(device)
        mode = mode.to(device)        
        batch = batch.to(device)
        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        rna = torch.tensor([0., 1.]).to(device)
        
        with pyro.plate('data'):
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (1.9 * self.output_size)
            y = pyro.sample('y', dist.OneHotCategorical(alpha_prior), obs=ys)
            
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1)).to(device)
    
            zy_loc, zy_scale = self.decoder_z2zy([zs, y])
            zys = pyro.sample("zy", dist.Normal(zy_loc, zy_scale).to_event(1))

            if not torch.equal(set(mode).pop(), rna):
                # ATAC数据处理
                ls_loc = torch.ones(batch_size, self.atac_input_size, **options)
                ls_scale = torch.ones(batch_size, self.atac_input_size, **options)
                ls = pyro.sample("atac_ls", dist.LogNormal(ls_loc, ls_scale).to_event(1))
            
                theta = pyro.param("inverse_dispersion_atac", 10.0 * xs.new_ones(self.atac_input_size),
                                 constraint=constraints.positive)
                gate_logits, mu = self.decoder_atac_thetas(zys)
                nb_logits = torch.log(ls * mu + self.epsilon) - torch.log(theta + self.epsilon)
                nb_logits = torch.clamp(nb_logits, min=-10, max=10)

                pyro.sample(
                    "obs_atac",
                    dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits
                    ).to_event(1),
                    obs=xs.int()
                ).to(device)
           
            if torch.equal(set(mode).pop(), rna):
                # RNA数据处理
                sum_per_sample = xs.sum(dim=1)
                if (sum_per_sample <= 0).any():
                    print("存在样本总和为零或负数！")
                    
                ls_loc = torch.ones(batch_size, self.rna_input_size, **options)
                ls_scale = torch.ones(batch_size, self.rna_input_size, **options)
                ls = pyro.sample("rna_ls", dist.LogNormal(ls_loc, ls_scale).to_event(1))
            
                theta = pyro.param("inverse_dispersion_rna", 10.0 * xs.new_ones(self.rna_input_size),
                                 constraint=constraints.positive)
                gate_logits, mu = self.decoder_rna_thetas(zys)
                nb_logits = torch.log(ls * mu + self.epsilon) - torch.log(theta + self.epsilon)
                nb_logits = torch.clamp(nb_logits, min=-10, max=10) 
               
                pyro.sample(
                    "obs_rna",
                    dist.ZeroInflatedNegativeBinomial(
                        gate_logits=gate_logits,
                        total_count=theta,
                        logits=nb_logits
                    ).to_event(1),
                    obs=xs.int()
                ).to(device)

    def guide(self, xs, ys=None, mode=torch.tensor([1., 0.]), batch=None):
        """
        原有的推理网络，保持不变
        """
        xs = xs.to(device)
        mode = mode.to(device)
        rna = torch.tensor([0., 1.]).to(device)
        batch_size = xs.size(0)
        
        with pyro.plate('data'):           
            if not torch.equal(set(mode).pop(), rna):
                zy_loc, zy_scale = self.encoder_atac2z(xs)
                zys = pyro.sample('zy', dist.Normal(zy_loc, zy_scale).to_event(1))           
                ls_loc, ls_scale = self.encoder_atac_ls(xs)
                pyro.sample("atac_ls", dist.LogNormal(ls_loc, ls_scale).to_event(1))
                
            if torch.equal(set(mode).pop(), rna):
                zy_loc, zy_scale = self.encoder_rna2z(xs)
                zys = pyro.sample('zy', dist.Normal(zy_loc, zy_scale).to_event(1))      
                ls_loc, ls_scale = self.encoder_rna_ls(xs)
                pyro.sample("rna_ls", dist.LogNormal(ls_loc, ls_scale).to_event(1))
            
            palpha_y = self.encoder_z2y(zys)
            if ys is None:
                ys = pyro.sample('y', dist.OneHotCategorical(palpha_y))

            z_loc, z_scale = self.encoder_z([zys, ys])
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def model_classify(self, xs, ys=None, mode=None, batch=None):
        """
        增强的分类模型，包含类别权重功能
        """
        pyro.module('scc', self)
        mode = mode.to(device)
        xs = xs.to(device)
        rna = torch.tensor([0., 1.]).to(device)
        
        with pyro.plate('data'):
            # 统一编码器处理        
            if not torch.equal(set(mode).pop(), rna):
                zy_loc, _ = self.encoder_atac2z(xs)    
            if torch.equal(set(mode).pop(), rna):
                zy_loc, _ = self.encoder_rna2z(xs)       
            
            alpha_y = self.encoder_z2y(zy_loc)

            if ys is None:
                # 无标签数据处理
                with torch.no_grad():
                    # 分布对齐
                    alpha_y_aligned = self.dist_aligner.dist_align(probs_ulb=alpha_y)
                    
                    # 更新类别权重（基于伪标签）
                    if self.use_class_weights:
                        confidences = F.softmax(alpha_y, dim=1)
                        self.update_class_weights_with_pseudolabels(alpha_y_aligned, confidences)
                
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(alpha_y_aligned))

            else:
                # 有标签数据处理
                with torch.no_grad():
                    self.dist_aligner.update(probs_lb=alpha_y)
                    
                    # 更新有标签数据的类别计数
                    if self.use_class_weights:
                        target_indices = ys.argmax(dim=1)
                        for idx in target_indices:
                            self.labeled_class_counts[idx.item()] += 1
                
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    if self.use_class_weights and self.class_weights_tensor is not None:
                        target_indices = ys.argmax(dim=1)
                        weights = self.class_weights_tensor[target_indices]
                        
                        # 使用factor方式应用个体权重
                        base_dist = dist.OneHotCategorical(alpha_y)
                        log_probs = base_dist.log_prob(ys)
                        
                        # 每个样本的权重单独应用
                        additional_weight_term = (weights - 1.0) * log_probs
                        pyro.factor('sample_weights', additional_weight_term.sum())
                        
                        # 正常采样（权重已通过factor应用）
                        ys_aux = pyro.sample('y_aux', base_dist, obs=ys)
                    else:
                        ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(alpha_y), obs=ys)
                

    def guide_classify(self, xs, ys=None, mode=None, batch=None):
        """
        分类的虚拟guide函数
        """
        pass

    def model_classify1(self, xs, ys=None, mode=None, batch=None, adv_training=True):
        """
        对抗训练模型，保持原有功能
        """
        pyro.module('scc', self)
        mode = mode.to(device)
        xs = xs.to(device)
        rna = torch.tensor([0., 1.]).to(device)
        
        with pyro.plate('data'):
            if not torch.equal(set(mode).pop(), rna):
                zy_loc, _ = self.encoder_atac2z(xs)
            else:
                zy_loc, _ = self.encoder_rna2z(xs)
 
            if batch is not None:
                batch_pred = self.batch_adv(zy_loc, self.adversarial_alpha)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample('batch_pred', dist.OneHotCategorical(batch_pred), obs=batch)

    def guide_classify1(self, xs, ys=None, batch=None, mode=None):
        """
        对抗训练的虚拟guide函数
        """
        pass

    def classifier(self, xs, mode=torch.tensor([1., 0.])):
        """
        分类器，保持原有功能
        """
        rna = torch.tensor([0., 1.])
        if not torch.equal(set(mode).pop(), rna):
            z_loc, _ = self.encoder_atac2z(xs)    
        if torch.equal(set(mode).pop(), rna):
            z_loc, _ = self.encoder_rna2z(xs)   
             
        alpha = self.encoder_z2y(z_loc)
        res, ind = torch.topk(alpha, 1)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def classifier_with_probability(self, xs, mode=torch.tensor([1., 0.])):
        """
        带概率的分类器，保持原有功能
        """
        rna = torch.tensor([0., 1.])
        if not torch.equal(set(mode).pop(), rna):
            z_loc, _ = self.encoder_atac2z(xs)    
        elif torch.equal(set(mode).pop(), rna):
            z_loc, _ = self.encoder_rna2z(xs)   
            
        alpha = self.encoder_z2y(z_loc)
        res, ind = torch.topk(alpha, 1)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys, alpha

    def latent_embedding(self, xs, mode=torch.tensor([1., 0.])):
        """
        潜在嵌入，保持原有功能
        """
        rna = torch.tensor([0., 1.])
        if torch.equal(set(mode).pop(), rna):
            zy, _ = self.encoder_rna2z(xs)   
            alpha = self.encoder_z2y(zy)
            res, ind = torch.topk(alpha, 1)
            ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
            z_loc, _ = self.encoder_z([zy, ys])
        elif not torch.equal(set(mode).pop(), rna):
            zy, _ = self.encoder_atac2z(xs)        
            alpha = self.encoder_z2y(zy)
            res, ind = torch.topk(alpha, 1)
            ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
            z_loc, _ = self.encoder_z([zy, ys])
        return z_loc
    def latent_embedding_zy(self, xs,mode=torch.tensor([1., 0.])):
        """
        compute the latent embedding of a cell (or a batch of cells)

        :param xs: a batch of vectors of gene counts from a cell
        :return: a batch of the latent embeddings
        """
        rna = torch.tensor([0., 1.])
        if torch.equal(set(mode).pop(), rna):
            zy, _= self.encoder_rna2z(xs)   
        elif not torch.equal(set(mode).pop(), rna):
            zy, _= self.encoder_atac2z(xs)        
        return zy
    