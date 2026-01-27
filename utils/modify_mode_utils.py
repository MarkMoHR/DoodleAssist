class ModifyMode:
    ADD_ONLY = 0
    EDIT_ONLY = 1
    DELETE_ONLY = 2
    ADD_EDIT = 3
    ADD_DELETE = 4
    EDIT_DELETE = 5
    ADD_EDIT_DELETE = 6
    UNCHANGED = 7

    mode_dict = {0: 'ADD_ONLY', 1: 'EDIT_ONLY', 2: 'DELETE_ONLY', 3: 'ADD_EDIT',
                 4: 'ADD_DELETE', 5: 'EDIT_DELETE', 6: 'ADD_EDIT_DELETE', 7: 'UNCHANGED'}


def detect_mode(modified_path_ids):
    len_add = len(modified_path_ids['add'])
    len_edit = len(modified_path_ids['edit'])
    len_delete = len(modified_path_ids['delete'])
    if len_add > 0 and len_edit == 0 and len_delete == 0:
        return ModifyMode.ADD_ONLY
    elif len_add == 0 and len_edit > 0 and len_delete == 0:
        return ModifyMode.EDIT_ONLY
    elif len_add == 0 and len_edit == 0 and len_delete > 0:
        return ModifyMode.DELETE_ONLY
    elif len_add > 0 and len_edit > 0 and len_delete == 0:
        return ModifyMode.ADD_EDIT
    elif len_add > 0 and len_edit == 0 and len_delete > 0:
        return ModifyMode.ADD_DELETE
    elif len_add == 0 and len_edit > 0 and len_delete > 0:
        return ModifyMode.EDIT_DELETE
    elif len_add > 0 and len_edit > 0 and len_delete > 0:
        return ModifyMode.ADD_EDIT_DELETE
    else:
        return ModifyMode.UNCHANGED
        # raise Exception('Invalid ModifyMode.')
