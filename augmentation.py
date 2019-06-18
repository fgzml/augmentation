def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
        ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def shift(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0., order=1):
    h, w = x.shape[row_index], x.shape[col_index]
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        tx, ty = hrg * h, wrg * w
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(
        x,
        transform_matrix,
        channel_index,
        fill_mode,
        cval,
        order)
    return x


def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    h, w = x.shape[row_index], x.shape[col_index]
    assert (
        h > hrg) and (
        w > wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        # print(h_offset, w_offset, x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].shape)
        return x[h_offset:hrg + h_offset, w_offset:wrg + w_offset]
    else:  # central crop
        h_offset = int(np.floor((h - hrg) / 2.))
        w_offset = int(np.floor((w - wrg) / 2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset:h_end, w_offset:w_end]


def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0., order=1):
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(
        x,
        transform_matrix,
        channel_index,
        fill_mode,
        cval,
        order)
    return x


def flip_axis(x, axis=1, is_random=False):
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
