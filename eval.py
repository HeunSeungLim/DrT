def testing_fun(model, test_loaders, args):
    model.eval()
    total_psnr_linear = 0
    total_psnr_mu = 0
    num = 0
    for data, target in test_loaders:
        Test_Data_name = test_loaders.dataset.img_list[num].split('.h5')[0].split('/')[-1]
        with torch.no_grad():

            data = data.transpose(2, 3)
            target = target.transpose(2, 3)

            input_patch = get_patch(data,500)
            target_patch = get_patch(target,500)
            psnr_linear, psnr_mu = 0, 0
            for i in range(len(input_patch)):
                data1 = torch.cat((input_patch[i][:,0:3,:,:],input_patch[i][:,9:12,:,:]),1)
                data2 = torch.cat((input_patch[i][:, 3:6, :, :], input_patch[i][:, 12:15, :, :]), 1)
                data3 = torch.cat((input_patch[i][:, 6:9, :, :], input_patch[i][:, 15:18, :, :]), 1)
                output_patch = model(data1.cuda(), data2.cuda(), data3.cuda())
                output_patch = output_patch.cpu()
                psnr_linear += psnr(output_patch, target_patch[i])
                psnr_mu += psnr(range_compressor(output_patch),range_compressor(target_patch[i]))
            psnr_linear /= len(input_patch)
            psnr_mu /= len(input_patch)

########################################
            # data, target = data.cuda(), target.cuda()



        # data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
        # data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
        # data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
        # data1 = Variable(data1, volatile=True)
        # data2 = Variable(data2, volatile=True)
        # data3 = Variable(data3, volatile=True)
        # target = Variable(target, volatile=True)
        # output = model(data1, data2, data3)

        # val = psnr(output,target)
        # mu_val = psnr(range_compressor(output),range_compressor(target))
        # save the result to .H5 files

        # output = torch.squeeze(output)
        # target = torch.squeeze(target)
        # output = output.cpu()
        # target = target.cpu()
        # output = output.detach().numpy()
        # target = target.detach().numpy()
        # output = np.rollaxis(output, 0, start=3)
        # target = np.rollaxis(output, 0, start=3)
        # imageio.imsave('./result/' + Test_Data_name + '_hdr.hdr', output, format='hdr')


        # hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
        # img = output[0, :, :, :]
        # img = tv.utils.make_grid(img.data.cpu()).numpy()
        # hdrfile.create_dataset('data', data=img)
        # hdrfile.close()

        # hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        # target = torch.log(1 + 5000 * target).cpu() / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        # test_loss += F.mse_loss(hdr, target)
        # num = num + 1
        # psnr = cv2.PSNR(output,target)
        # mu = cv2.PSNR(log((1 + 5000*output))/log(1+5000),log((1 + 5000*target))/log(1+5000))
        total_psnr_linear += psnr_linear
        total_psnr_mu += psnr_mu
    avg_psnr_linear = total_psnr_linear / len(test_loaders.dataset)
    avg_psnr_mu = total_psnr_mu/ len(test_loaders.dataset)
    print('\n l-psnr:' + str(avg_psnr_linear))
    print('\n mu-psnr:' + str(avg_psnr_mu))
    return avg_psnr_mu