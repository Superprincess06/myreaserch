import torch.nn as nn


class MyNet(nn.Module):
	"""A lightweight placeholder model for future custom architecture experiments."""

	def __init__(self, num_classes: int) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(1),
		)
		self.classifier = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.features(x)
		x = x.flatten(1)
		return self.classifier(x)
