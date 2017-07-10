
------------------------------
-- function
------------------------------

function split_valid(all_list)

	local train_list = {}
	local valid_list = {}
	local shuffle = torch.randperm(#all_list)

	local num_valid = math.floor(#all_list / 10)

	for i = 1,num_valid do
		table.insert(valid_list, all_list[shuffle[i]])
	end

	for i = num_valid+1,#all_list do
		table.insert(train_list, all_list[shuffle[i]])
	end

	table.sort(train_list, function(a,b) return (tonumber(a) < tonumber(b)) end)
	table.sort(valid_list, function(a,b) return (tonumber(a) < tonumber(b)) end)

	assert(#train_list + #valid_list == #all_list, 'train + valid != all')

	print('\tnum train: ' .. #train_list)
	print('\t\t' .. tostring(tablex.sub(train_list,1,10)) .. '...'
		.. tostring(tablex.sub(train_list,-10,Inf)))
	print('\tnum valid: ' .. #valid_list)
	print('\t\t' .. tostring(tablex.sub(valid_list,1,10)) .. '...'
		.. tostring(tablex.sub(valid_list,-10,Inf)))

	return train_list, valid_list
end
