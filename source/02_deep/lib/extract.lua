
------------------------------
-- function
------------------------------

function extract(img)

	local f1 = torch.sum(img)
	local f2 = torch.sum(torch.pow(img, 2))

	return torch.Tensor({1,f1,f2})
end
