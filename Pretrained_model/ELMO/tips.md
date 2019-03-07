char_id: 0-255 utf-8
beginning_of_sentence_character = 256  # <begin sentence>
end_of_sentence_character = 257  # <end sentence>
beginning_of_word_character = 258  # <begin word>
end_of_word_character = 259  # <end word>
padding_character = 260 # <padding>

beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length
)
end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length
)

def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids

convolutions = []
for i, (width, num) in enumerate(filters):
    conv = torch.nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=num,
            kernel_size=width,
            bias=True
    )

{'char_cnn': {
    'activation': 'relu',
    'embedding': {'dim': 4},
    'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
    'max_characters_per_token': 50,
    'n_characters': 262,
    'n_highway': 2
    }
}

padding for sentence:0
padding for character:261
因为0被padding for sentence占据，所以utf-8 0-255的映射包括特殊字符都向后退一位
character层采用textcnn处理，多个卷积然后取最大值，concat在一起即可
上层2层bilstm，预计采用pytorch官方实现，就不自己重写。

双向lstm训练时候，因为要屏蔽当前单词。假设当前单词下标为k，总长度为n，正向lstm只取1到(k-1)，反向取n到(k+1)。
实现的话，正向和反向分开写，也是分开训练