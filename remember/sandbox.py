linked_list = __import__('1_SingleLinkedList')


def sum_lists(list1_head, list2_head):
    if list1_head is None and list2_head is None:
        return None
    elif list1_head is None:
        return list2_head
    elif list2_head is None:
        return list1_head

    pointer1 = list1_head
    pointer2 = list2_head

    current_sum = pointer1.value + pointer2.value
    is_carry = current_sum >= 10

    head = linked_list.SingleNode(current_sum % 10)
    current_node = head

    pointer1 = pointer1.next
    pointer2 = pointer2.next

    while pointer1 is not None or pointer2 is not None:
        current_sum = (1 if is_carry else 0)
        current_sum += pointer1.value if pointer1 is not None else 0
        current_sum += pointer2.value if pointer2 is not None else 0

        is_carry = current_sum >= 10
        current_node.next = linked_list.SingleNode(current_sum % 10)

        if pointer1 is not None:
            pointer1 = pointer1.next
        if pointer2 is not None:
            pointer2 = pointer2.next

        current_node = current_node.next

    return head


head1 = linked_list.construct_linked_list([7, 1, 6])
head2 = linked_list.construct_linked_list([5, 9, 2])
head = sum_lists(head1, head2)
linked_list.print_linked_list(head)
