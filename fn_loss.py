import torch
from torch import nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
	

class RBF(nn.Module):

	def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
		super().__init__()
		self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
		self.bandwidth = bandwidth

	def get_bandwidth(self, L2_distances):
		if self.bandwidth is None:
			n_samples = L2_distances.shape[0]
			return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

		return self.bandwidth

	def forward(self, X):
		L2_distances = torch.cdist(X, X) ** 2
		L2_distances = L2_distances.to(X.device)
		return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

	def __init__(self, kernel=RBF()):
		super().__init__()
		self.kernel = kernel

	def forward(self, X, Y):
		# self.kernel = self.kernel.to(X.device)
		K = self.kernel(torch.vstack([X, Y]))
		# K = K.to(X.device)

		X_size = X.shape[0]
		XX = K[:X_size, :X_size].mean()
		XY = K[:X_size, X_size:].mean()
		YY = K[X_size:, X_size:].mean()
		return XX - 2 * XY + YY


def MMD(x, y, kernel="rbf", device="cpu"): 
	"""Emprical maximum mean discrepancy. The lower the result
	   the more evidence that distributions are the same.
	   https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py

	Args:
		x: first sample, distribution P
		y: second sample, distribution Q
		kernel: kernel type such as "multiscale" or "rbf"
	"""
	xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))
	
	dxx = rx.t() + rx - 2. * xx # Used for A in (1)
	dyy = ry.t() + ry - 2. * yy # Used for B in (1)
	dxy = rx.t() + ry - 2. * zz # Used for C in (1)
	
	XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
				  torch.zeros(xx.shape).to(x.device),
				  torch.zeros(xx.shape).to(x.device))
	# XX, YY, XY = (torch.zeros(xx.shape).to(device),
	# 			  torch.zeros(xx.shape).to(device),
	# 			  torch.zeros(xx.shape).to(device))
	if kernel == "multiscale":
		
		bandwidth_range = [0.2, 0.5, 0.9, 1.3]
		for a in bandwidth_range:
			XX += a**2 * (a**2 + dxx)**-1
			YY += a**2 * (a**2 + dyy)**-1
			XY += a**2 * (a**2 + dxy)**-1
			
	if kernel == "rbf":
	  
		bandwidth_range = [10, 15, 20, 50]
		for a in bandwidth_range:
			XX += torch.exp(-0.5*dxx/a)
			YY += torch.exp(-0.5*dyy/a)
			XY += torch.exp(-0.5*dxy/a)

	return torch.mean(XX + YY - 2. * XY)


class MMD_loss(nn.Module):
	r"""Maximum Mean Discrepancy loss.

	Ref: https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
	"""
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	 
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)

		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2) 
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss
	

class TripletLoss(nn.Module):
	def __init__(self, margin=1.0, is_euclid=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.is_euclid = is_euclid

	def get_name(self):
		return f"({self.__class__.__name__}, margin:{self.margin}, euclid:{self.is_euclid})"

	def calc_euclidean(self, x1, x2):
		return (x1 - x2).pow(2).sum(1)

	def calc_mmd(self, x1, x2):
		return MMD(x1, x2)

	def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
		if self.is_euclid:
			distance_positive = self.calc_euclidean(anchor, positive)
			distance_negative = self.calc_euclidean(anchor, negative)
		else:
			distance_positive = self.calc_mmd(anchor, positive)
			distance_negative = self.calc_mmd(anchor, negative)
		losses = torch.relu(distance_positive - distance_negative + self.margin)
		return losses.mean()
	

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
			target.float() * distances 
			+ (1 + -1 * target  ).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()